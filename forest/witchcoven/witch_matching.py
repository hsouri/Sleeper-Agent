"""Main class, holding information about models and training/testing routines."""

import random
import torch
import random
import torchvision
from PIL import Image
from ..consts import BENCHMARK, NON_BLOCKING
from ..utils import bypass_last_layer
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK
from .witch_base import _Witch

class WitchGradientMatching(_Witch):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)
            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * poison_loss
            passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _passenger_loss(self, poison_grad, source_grad, source_clean_grad, source_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 5)
        else:
            indices = torch.arange(len(source_grad))


        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (source_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (source_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])

            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                poison_norm += poison_grad[i].pow(2).sum()

        if self.args.repel != 0:
            for i in indices:
                if self.args.loss in ['scalar_product', *SIM_TYPE]:
                    passenger_loss += self.args.repel * (source_grad[i] * poison_grad[i]).sum()
                elif self.args.loss == 'cosine1':
                    passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                elif self.args.loss == 'SE':
                    passenger_loss -= 0.5 * self.args.repel * (source_grad[i] - poison_grad[i]).pow(2).sum()
                elif self.args.loss == 'MSE':
                    passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])

        passenger_loss = passenger_loss / source_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * poison_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / source_gnorm

        return passenger_loss
    
    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch
        
    def patch_sources(self, kettle):
        if self.args.load_patch == '':
            patch = self._create_patch([3, int(self.args.patch_size), int(self.args.patch_size)])
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.patch_size))
            patch = totensor(resize(patch))

        patch = (patch.to(**self.setup) - kettle.dm) / kettle.ds
        self.patch = patch.squeeze(0)

        

        # Add patch to sourceset
        if self.args.random_patch:
            print("Add patches to the source images randomely ...")
        else:
            print("Add patches to the source images on the bottom right ...")

        source_delta = []
        for idx, (source_img, label, image_id) in enumerate(kettle.sourceset):
            source_img = source_img.to(**self.setup)

            if self.args.random_patch:
                patch_x = random.randrange(0,source_img.shape[1] - self.patch.shape[1] + 1)
                patch_y = random.randrange(0,source_img.shape[2] - self.patch.shape[2] + 1)
            else:
                patch_x = source_img.shape[1] - self.patch.shape[1]
                patch_y = source_img.shape[2] - self.patch.shape[2]

            delta_slice = torch.zeros_like(source_img).squeeze(0)
            diff_patch = self.patch - source_img[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]]
            delta_slice[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]] = diff_patch
            source_delta.append(delta_slice.cpu())
        kettle.sourceset = datasets.Deltaset(kettle.sourceset, source_delta)

        # Add patch to sourceset
        if self.args.random_patch:
            print("Add patches to the source train images randomely ...")
        else:
            print("Add patches to the source train images on the bottom right ...")

        source_delta = []
        for idx, (source_img, label, image_id) in enumerate(kettle.source_trainset):
            source_img = source_img.to(**self.setup)

            if self.args.random_patch:
                patch_x = random.randrange(0,source_img.shape[1] - self.patch.shape[1] + 1)
                patch_y = random.randrange(0,source_img.shape[2] - self.patch.shape[2] + 1)
            else:
                patch_x = source_img.shape[1] - self.patch.shape[1]
                patch_y = source_img.shape[2] - self.patch.shape[2]

            delta_slice = torch.zeros_like(source_img).squeeze(0)
            diff_patch = self.patch - source_img[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]]
            delta_slice[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]] = diff_patch
            source_delta.append(delta_slice.cpu())
        kettle.source_trainset = datasets.Deltaset(kettle.source_trainset, source_delta)




class WitchGradientMatchingNoisy(WitchGradientMatching):
    """Brew passenger poison with given arguments.

    Both the poison gradient and the source gradient are modified to be diff. private before calcuating the loss.
    """

    def _initialize_brew(self, victim, kettle):
        super()._initialize_brew(victim, kettle)
        self.defs = victim.defs

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True, only_inputs=True)

            # add noise to samples
            self._hide_gradient(poison_grad)

            # Compute blind passenger loss
            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)
            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * poison_loss
            passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _hide_gradient(self, gradient_list):
        """Enforce batch-wise privacy if necessary.

        This is attacking a defense discussed in Hong et al., 2020
        We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        of noise to the gradient signal
        """
        if self.defs.privacy['clip'] is not None:
            total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradient_list]))
            clip_coef = self.defs.privacy['clip'] / (total_norm + 1e-6)
            if clip_coef < 1:
                for grad in gradient_list:
                    grad.mul(clip_coef)

        if self.defs.privacy['noise'] is not None:
            loc = torch.as_tensor(0.0, device=kettle.setup['device'])
            clip_factor = defs.privacy['clip'] if defs.privacy['clip'] is not None else 1.0
            scale = torch.as_tensor(clip_factor * defs.privacy['noise'], device=kettle.setup['device'])
            if self.defs.privacy['distribution'] == 'gaussian':
                generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif self.defs.privacy['distribution'] == 'laplacian':
                generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {self.defs.privacy["distribution"]} given.')

            for grad in gradient_list:
                grad += generator.sample(grad.shape)



class WitchGradientMatchingHidden(WitchGradientMatching):
    """Brew passenger poison with given arguments.

    Try to match the original image feature representation to hide the attack from filter defenses.
    This class does a ton of horrid overwriting of the _batched_step method to add some additional
    computations that I dont want to be executed for all attacks. todo: refactor :>
    """
    FEATURE_WEIGHT = 1.0

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current source loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        # save out clean inputs
        # These will be representative of "average" unpoisoned versions of the poison images
        # as such they will be augmented differently
        clean_inputs = inputs.clone().detach()

        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Add additional clean data if mixing during the attack:
            if self.args.pmix:
                if 'mix' in victim.defs.mixing_method['type']:   # this covers mixup, cutmix 4waymixup, maxup-mixup
                    try:
                        extra_data = next(self.extra_data)
                    except StopIteration:
                        self.extra_data = iter(kettle.trainloader)
                        extra_data = next(self.extra_data)
                    extra_inputs = extra_data[0].to(**self.setup)
                    extra_labels = extra_data[1].to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                    inputs = torch.cat((inputs, extra_inputs), dim=0)
                    clean_inputs = torch.cat((clean_inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs, randgen=randgen)
                clean_inputs = kettle.augment(clean_inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)
                clean_inputs, _, _ = kettle.mixer(clean_inputs, labels)

            if self.args.padversarial is not None:
                # The optimal choice of the 3rd and 4th argument here are debatable
                # This is likely the strongest anti-defense:
                # but the defense itself splits the batch and uses half of it as sources
                # instead of using the known source [as the defense does not know about the source]
                # delta = self.attacker.attack(inputs.detach(), labels,
                #                              self.sources, self.true_classes, steps=victim.defs.novel_defense['steps'])

                # This is a more accurate anti-defense:
                [temp_sources, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, source_selection=victim.defs.novel_defense['source_selection'])
                delta, additional_info = self.attacker.attack(inputs.detach(), labels,
                                                              temp_sources, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta  # Kind of a reparametrization trick
                clean_inputs = clean_inputs + delta



            # Define the loss objective and compute gradients
            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            closure = self._define_objective(inputs, clean_inputs, labels, criterion, self.sources, self.target_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.source_grad, self.source_clean_grad, self.source_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()


    def _define_objective(self, inputs, clean_inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            feature_model, last_layer = bypass_last_layer(model)
            features = feature_model(inputs)
            outputs = last_layer(features)

            # clean features:
            clean_features = feature_model(clean_inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True, only_inputs=True)

            # add feature term
            feature_loss = (features - clean_features).pow(2).mean()

            # Compute blind passenger loss
            passenger_loss = self._passenger_loss(poison_grad, source_grad, source_clean_grad, source_gnorm)

            total_loss = passenger_loss + self.FEATURE_WEIGHT * feature_loss
            if self.args.centreg != 0:
                total_loss = passenger_loss + self.args.centreg * poison_loss
            total_loss.backward(retain_graph=self.retain)
            return total_loss.detach().cpu(), prediction.detach().cpu()
        return closure




class WitchMatchingMultiSource(WitchGradientMatching):
    """Variant in which source gradients are matched separately."""

    def _initialize_brew(self, victim, kettle):
        super()._initialize_brew(victim, kettle)
        self.source_grad, self.source_gnorm = [], []
        for source, target_class in zip(self.sources, self.target_classes):
            grad, gnorm = victim.gradient(source.unsqueeze(0), target_class.unsqueeze(0))
            self.source_grad.append(grad)
            self.source_gnorm.append(gnorm)


    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            differentiable_params = [p for p in model.parameters() if p.requires_grad]
            outputs = model(inputs)

            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

            matching_loss = 0
            for tgrad, tnorm in zip(source_grad, source_gnorm):
                matching_loss += self._passenger_loss(poison_grad, tgrad, None, tnorm)
            if self.args.centreg != 0:
                matching_loss = matching_loss + self.args.centreg * poison_loss
            matching_loss.backward(retain_graph=self.retain)
            return matching_loss.detach().cpu(), prediction.detach().cpu()
        return closure
