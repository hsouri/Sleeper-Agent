"""Utilites related to training models."""


def print_and_save_stats(epoch, stats, current_lr, train_loss, train_acc, predictions, valid_loss,
                         source_acc, source_loss, source_clean_acc, source_clean_loss):
    """Print info into console and into the stats object."""
    stats['train_losses'].append(train_loss)
    stats['train_accs'].append(train_acc)

    if predictions is not None:
        stats['valid_accs'].append(predictions['all']['avg'])
        stats['valid_accs_target'].append(predictions['target']['avg'])
        stats['valid_accs_source'].append(predictions['source']['avg'])
        stats['valid_losses'].append(valid_loss)

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
              f'Validation   loss is {stats["valid_losses"][-1]:7.4f}, valid acc: {stats["valid_accs"][-1]:7.2%} | ')

        stats['source_accs'].append(source_acc)
        stats['source_losses'].append(source_loss)
        stats['source_accs_clean'].append(source_clean_acc)
        stats['source_losses_clean'].append(source_clean_loss)
        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Source adv. loss is {source_loss:7.4f}, fool  acc: {source_acc:7.2%} | '
              f'Source orig. loss is {source_clean_loss:7.4f}, orig. acc: {source_clean_acc:7.2%} | ')

    else:
        if 'valid_accs' in stats:
            # Repeat previous answers if validation is not recomputed
            stats['valid_accs'].append(stats['valid_accs'][-1])
            stats['valid_accs_target'].append(stats['valid_accs_target'][-1])
            stats['valid_accs_source'].append(stats['valid_accs_source'][-1])
            stats['valid_losses'].append(stats['valid_losses'][-1])
            stats['source_accs'].append(stats['source_accs'][-1])
            stats['source_losses'].append(stats['source_losses'][-1])
            stats['source_accs_clean'].append(stats['source_accs_clean'][-1])
            stats['source_losses_clean'].append(stats['source_losses_clean'][-1])

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | ')
