"""Implement batch-level attack steps."""

import torch
import higher
import random

from ..utils import _gradient_matching, bypass_last_layer


def construct_attack(novel_defense, model, loss_fn, dm, ds, tau, init, optim, num_classes, setup):
    """Interface for this submodule."""
    eps = novel_defense['strength']  # The defense parameter encodes the eps bound used during training
    if 'adversarial-evasion' in novel_defense['type']:
        return AdversarialAttack(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-wb' in novel_defense['type']:
        return AlignmentPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-se' in novel_defense['type']:
        return MatchingPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-mp' in novel_defense['type']:
        return MetaPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-fc' in novel_defense['type'] or 'adversarial-cp' in novel_defense['type']:
        return FeatureCollisionPoisoning(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-random' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, init, optim, num_classes, setup)
    elif 'adversarial-laplacian' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, 'laplacian', optim, num_classes, setup)
    elif 'adversarial-bernoulli' in novel_defense['type']:
        return RandomAttack(model, loss_fn, dm, ds, tau, eps, 'bernoulli', optim, num_classes, setup)
    elif 'adversarial-watermark' in novel_defense['type']:
        return WatermarkPoisoning(model, loss_fn, dm, ds, setup=setup)
    elif 'adversarial-patch' in novel_defense['type']:
        return PatchAttack(model, loss_fn, dm, ds, setup=setup)
    elif 'adversarial-htbd' in novel_defense['type']:
        return HTBD(model, loss_fn, dm, ds, setup=setup)
    else:
        raise ValueError(f'Invalid adversarial training objective specified: {novel_defense["type"]}.')


class BaseAttack(torch.nn.Module):
    """Implement a variety of input-altering attacks."""

    def __init__(self, model, loss_fn, dm=(0, 0, 0), ds=(1, 1, 1), tau=0.1, eps=16, init='zero',
                 optim='signAdam', num_classes=10, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with dict containing type and strength of attack and model info."""
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.setup = setup
        self.num_classes = num_classes

        self.dm, self.ds, = dm, ds
        self.tau, self.eps = tau, eps
        self.bound = self.eps / self.ds / 255

        self.init = init
        self.optim = optim

    def attack(self, inputs, labels, temp_sources, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)
        optimizer = self._init_optimizer(delta)

        for step in range(steps):
            optimizer.zero_grad()
            # Gradient step
            loss = self._objective(inputs + delta, labels, temp_sources, temp_fake_labels)
            delta.grad, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, only_inputs=True)
            # Optim step
            if 'sign' in self.optim:
                delta.grad.sign_()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                delta.data = torch.max(torch.min(delta, self.bound), -self.bound)
                delta.data = torch.max(torch.min(delta, (1 - self.dm) / self.ds - inputs), - self.dm / self.ds - inputs)

        delta.requires_grad = False
        additional_info = None
        return delta, additional_info

    def _objective(self, inputs, labels, temp_sources, temp_fake_labels):
        raise NotImplementedError()

    def _init_perturbation(self, input_shape):
        if self.init == 'zero':
            delta = torch.zeros(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
        elif self.init == 'rand':
            delta = (torch.rand(input_shape, device=self.setup['device'], dtype=self.setup['dtype']) - 0.5) * 2
            delta *= self.eps / self.ds / 255
        elif self.init == 'bernoulli':
            delta = (torch.rand(input_shape, device=self.setup['device'], dtype=self.setup['dtype']) > 0.5).float() * 2 - 1
            delta *= self.eps / self.ds / 255
        elif self.init == 'randn':
            delta = torch.randn(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
            delta *= self.eps / self.ds / 255
        elif self.init == 'laplacian':
            loc = torch.as_tensor(0.0, device=self.setup['device'])
            scale = torch.as_tensor(self.eps / self.ds / 255, device=self.setup['device']).mean()
            generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            delta = generator.sample(input_shape)
        elif self.init == 'normal':
            delta = torch.randn(input_shape, device=self.setup['device'], dtype=self.setup['dtype'])
        else:
            raise ValueError(f'Invalid init {self.init} given.')

        # Clamp initialization in all cases
        delta.data = torch.max(torch.min(delta, self.eps / self.ds / 255), -self.eps / self.ds / 255)
        delta.requires_grad_()
        return delta

    def _init_optimizer(self, delta):
        tau_sgd = (self.bound * self.tau).mean()
        if 'Adam' in self.optim:
            return torch.optim.Adam([delta], lr=self.tau, weight_decay=0)
        elif 'momSGD' in self.optim:
            return torch.optim.SGD([delta], lr=tau_sgd, momentum=0.9, weight_decay=0)
        else:
            return torch.optim.SGD([delta], lr=tau_sgd, momentum=0.0, weight_decay=0)


class AdversarialAttack(BaseAttack):
    """Implement a basic unsourceed attack objective."""

    def _objective(self, inputs, labels, temp_sources, temp_labels):
        """Evaluate negative CrossEntropy for a gradient ascent."""
        outputs = self.model(inputs)
        loss = -self.loss_fn(outputs, labels)
        return loss


class RandomAttack(BaseAttack):
    """Sanity check: do not actually attack - just use the random initialization."""


    def attack(self, inputs, labels, temp_sources, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)

        # skip optimization
        pass

        delta.requires_grad = False
        return delta, None


class WatermarkPoisoning(BaseAttack):
    """Sanity check: attack by watermarking."""

    def attack(self, inputs, labels, temp_sources, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective. This is effectively a slight mixing.

        with mixing factor lmb = 1 - eps / 255."""
        img_shape = temp_sources.shape[1:]
        num_sources = temp_sources.shape[0]
        num_inputs = inputs.shape[0]

        # Place
        if num_sources == num_inputs:
            delta = temp_sources - inputs
        elif num_sources < num_inputs:
            delta = temp_sources.repeat(num_inputs // num_sources, 1, 1, 1)[:num_inputs] - inputs
        else:
            factor = num_sources // num_inputs
            delta = temp_sources[:(factor * num_sources)].reshape(num_inputs, -1, *img_shape).mean(dim=1) - inputs
        delta *= self.eps / self.ds / 255

        return delta, None


class AlignmentPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via gradient alignment."""

    def _objective(self, inputs, labels, temp_sources, temp_fake_labels):
        """Evaluate Gradient Alignment and descend."""
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]

        poison_loss = self.loss_fn(self.model(inputs), labels)
        poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

        source_loss = self.loss_fn(self.model(temp_sources), temp_fake_labels)
        source_grad = torch.autograd.grad(source_loss, differentiable_params, retain_graph=True, create_graph=True)

        return _gradient_matching(poison_grad, source_grad)


class MatchingPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via gradient alignment."""

    def _objective(self, inputs, labels, temp_sources, temp_fake_labels):
        """Evaluate Gradient Alignment and descend."""
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]

        poison_loss = self.loss_fn(self.model(inputs), labels)
        poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

        source_loss = self.loss_fn(self.model(temp_sources), temp_fake_labels)
        source_grad = torch.autograd.grad(source_loss, differentiable_params, retain_graph=True, create_graph=True)

        objective, tnorm = 0, 0
        for pgrad, tgrad in zip(poison_grad, source_grad):
            objective += 0.5 * (tgrad - pgrad).pow(2).sum()
            tnorm += tgrad.detach().pow(2).sum()
        return objective / tnorm.sqrt()  # tgrad is a constant normalization factor as in witch_matching


class MetaPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via MetaPoison."""

    NADAPT = 2

    def _objective(self, inputs, labels, temp_sources, temp_fake_labels):
        """Evaluate Metapoison."""
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.1)
        # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
        with higher.innerloop_ctx(self.model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
            for _ in range(self.NADAPT):
                outputs = fmodel(inputs)
                poison_loss = self.loss_fn(outputs, labels)

                fopt.step(poison_loss)

        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        # model.eval()
        source_loss = self.loss_fn(fmodel(temp_sources), temp_fake_labels)
        return source_loss


class FeatureCollisionPoisoning(BaseAttack):
    """Implement limited steps for data poisoning via feature collision (with the bullseye polytope variant)."""

    def _objective(self, inputs, labels, temp_sources, temp_labels):
        """Evaluate Gradient Alignment and descend."""
        feature_model, last_layer = bypass_last_layer(self.model)

        # Get standard output:
        outputs = feature_model(inputs)
        outputs_sources = feature_model(temp_sources)

        return (outputs.mean(dim=0) - outputs_sources.mean(dim=0)).pow(2).mean()


class HTBD(BaseAttack):
    """Implement limited steps for data poisoning via hidden trigger backdoor.

    Note that this attack modifies temp_sources as a side-effect!"""

    def attack(self, inputs, labels, temp_sources, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        if delta is None:
            delta = self._init_perturbation(inputs.shape)
        optimizer = self._init_optimizer(delta)

        temp_sources = self._apply_patch(temp_sources)
        for step in range(steps):
            input_indcs, source_indcs = self._index_mapping(inputs, temp_sources)
            optimizer.zero_grad()
            # Gradient step
            loss = self._objective(inputs + delta, temp_sources, input_indcs, source_indcs)
            delta.grad, = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, only_inputs=True)
            # Optim step
            if 'sign' in self.optim:
                delta.grad.sign_()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                delta.data = torch.max(torch.min(delta, self.bound), -self.bound)
                delta.data = torch.max(torch.min(delta, (1 - self.dm) / self.ds - inputs), - self.dm / self.ds - inputs)

        delta.requires_grad = False
        return delta, None

    def _objective(self, inputs, temp_sources, input_indcs, source_indcs):
        """Evaluate Gradient Alignment and descend."""
        feature_model, last_layer = bypass_last_layer(self.model)
        new_inputs = torch.zeros_like(inputs)
        new_sources = torch.zeros_like(temp_sources)
        for i in range(len(input_indcs)):
            new_inputs[i] = inputs[input_indcs[i]]
            new_sources[i] = temp_sources[source_indcs[i]]

        outputs = feature_model(new_inputs)
        outputs_sources = feature_model(new_sources)
        return (outputs - outputs_sources).pow(2).mean(dim=1).sum()

    def _apply_patch(self, temp_sources):
        patch_shape = [[3, random.randint(int(0.2 * temp_sources.shape[2]), int(0.4 * temp_sources.shape[2])),
                        random.randint(int(0.1 * temp_sources.shape[3]), int(0.2 * temp_sources.shape[3]))] for _ in range(temp_sources.shape[0])]

        patch = self._create_patch(patch_shape)
        patch = [p.to(**self.setup) for p in patch]
        x_locations, y_locations = self._set_locations(temp_sources.shape, patch_shape)
        for i in range(len(patch)):
            temp_sources[i, :, x_locations[i]:x_locations[i] + patch_shape[i][1], y_locations[i]:y_locations[i]
                         + patch_shape[i][2]] = patch[i] - temp_sources[i, :, x_locations[i]:x_locations[i] + patch_shape[i][1],
                                                                        y_locations[i]:y_locations[i] + patch_shape[i][2]]
        return temp_sources

    def _index_mapping(self, inputs, temp_sources):
        with torch.no_grad():
            feature_model, last_layer = bypass_last_layer(self.model)
            feat_source = feature_model(inputs)
            feat_source = feature_model(temp_sources)
            dist = torch.cdist(feat_source, feat_source)
            input_indcs = []
            source_indcs = []
            for _ in range(feat_source.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(as_tuple=False).squeeze()
                input_indcs.append(dist_min_index[0])
                source_indcs.append(dist_min_index[1])
                dist[dist_min_index[0], dist_min_index[1]] = 1e5
        return input_indcs, source_indcs

    def _set_locations(self, input_shape, patch_shape):
        ''' fix locations where we’ll put the patches '''
        x_locations = []
        y_locations = []
        for i in range(input_shape[0]):
            x_locations.append(random.randint(0, input_shape[2] - patch_shape[i][1]))
            y_locations.append(random.randint(0, input_shape[3] - patch_shape[i][2]))
        return x_locations, y_locations

    def _create_patch(self, patch_shape):
        # create same patch or different one?
        patches = []
        for i in range(len(patch_shape)):
            temp_patch = 0.5 * torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            patch = torch.bernoulli(temp_patch)
            patches.append(patch)
        return patches


class PatchAttack(BaseAttack):
    """Randomly patch 2 classes."""

    def _objective(self, inputs, labels, temp_sources, temp_labels):
        """Evaluate negative CrossEntropy for a gradient ascent."""
        pass


    def attack(self, inputs, labels, temp_sources, temp_true_labels, temp_fake_labels, steps=5, delta=None):
        """Attack within given constraints with task as in _objective."""
        patch_shape = [[3, random.randint(int(0.1 * inputs.shape[2]), int(0.4 * inputs.shape[2])),
                        random.randint(int(0.1 * inputs.shape[3]), int(0.4 * inputs.shape[3]))] for _ in range(self.num_classes)]
        # patch_shape = [[3, self.eps, self.eps] for _ in range(self.num_classes)]

        x_locations, y_locations = self._set_locations(inputs.shape, labels, patch_shape)
        # Maybe different patch per class?
        patch = self._create_patch(patch_shape)

        if delta is None:
            delta1 = self._init_perturbation(inputs.shape)
            delta2 = self._init_perturbation(temp_sources.shape)
        delta1.requires_grad = False
        delta2.requires_grad = False

        for i in range(delta1.shape[0]):
            # Patch every class
            temp_label = labels[i]
            delta1[i, :, x_locations[i]:x_locations[i] + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                   + patch_shape[temp_label][2]] = patch[temp_label] - inputs[i, :, x_locations[i]:x_locations[i]
                                                                              + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                                                                              + patch_shape[temp_label][2]]

        # Maybe different patch per class?
        # patch = [self._create_patch(patch_shape).to(**self.setup) for _ in range(num_classes)]
        permute_list = self._random_derangement(self.num_classes)
        temp_source_labels = [permute_list[temp_true_label] for temp_true_label in temp_true_labels]
        x_locations, y_locations = self._set_locations(temp_sources.shape, temp_source_labels, patch_shape)
        for i in range(delta2.shape[0]):
            temp_label = permute_list[temp_true_labels[i]]
            delta2[i, :, x_locations[i]:x_locations[i] + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                   + patch_shape[temp_label][2]] = patch[temp_label] - temp_sources[i, :, x_locations[i]:x_locations[i]
                                                                                    + patch_shape[temp_label][1], y_locations[i]:y_locations[i]
                                                                                    + patch_shape[temp_label][2]]
        #
        return [delta1, delta2]

    def _set_locations(self, input_shape, labels, patch_shape):
        ''' fix locations where we’ll put the patches '''
        x_locations = []
        y_locations = []
        for i in range(input_shape[0]):
            x_locations.append(random.randint(0, input_shape[2] - patch_shape[labels[i]][1]))
            y_locations.append(random.randint(0, input_shape[3] - patch_shape[labels[i]][2]))
        return x_locations, y_locations

    def _create_patch(self, patch_shape):
        # create same patch or different one?
        patches = []
        for i in range(len(patch_shape)):
            param = random.random()
            # temp_patch = 0.5*torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            temp_patch = param * torch.ones(patch_shape[i][0], patch_shape[i][1], patch_shape[i][2])
            patch = torch.bernoulli(temp_patch)
            patches.append(patch.to(**self.setup) / self.ds)
        return patches

    def _random_derangement(self, n):
        while True:
            v = [i for i in range(n)]
            for j in range(n - 1, -1, -1):
                p = random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return v
