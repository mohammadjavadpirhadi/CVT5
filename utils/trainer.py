import os
import shutil
import json
from tqdm.auto import tqdm
import evaluate
import datasets
import torch
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as sn_evaluate

from utils.captioning import caption_logits_to_string, caption_ids_to_string

meteor_calculator = evaluate.load('meteor', download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS)

class AverageMeter:
    # source: https://github.com/antoyang/just-ask
    """ Computes and stores the average and current value for training stats """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def move_batch_to_device(batch, device):
    return {
        'indices': batch["indices"],
        'frames_map': batch["frames_map"].to(device),
        'temporal_masks': batch["temporal_masks"].to(device),
        'captioner_attention_masks': batch["captioner_attention_masks"].to(device),
        'I_frames': batch["I_frames"].to(device),
        'motion_vectors': batch["motion_vectors"].to(device),
        'residuals': batch["residuals"].to(device) if batch["residuals"] is not None else None,
        'transcripts_features': batch["transcripts_features"].to(device) if batch["transcripts_features"] is not None else None,
        'spotting_labels': batch["spotting_labels"].to(device) if batch["spotting_labels"] is not None else None,
        'captions': batch["captions"] if batch["captions"] is not None else None,
        'captions_input_ids': batch["captions_input_ids"].to(device) if batch["captions_input_ids"] is not None else None
    }


def train_one_epoch(model, data_loader, short_memory_len, umt5_enabled, exclude_bg, epoch_index, optimizer, scheduler, tb_writer, device, verbose=0):
    loss_metric = AverageMeter()
    accuracy_metric = BinaryAccuracy()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    f1_metric = BinaryF1Score()
    meteor_metric = AverageMeter()

    progress_bar = tqdm(enumerate(data_loader), desc=f"Train epoch {epoch_index+1}", total=len(data_loader))
    for i, batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        spotting_logits, caption_logits, step_loss = model(**batch)
        spotting_predictions = torch.sigmoid(spotting_logits)

        loss_metric.update(step_loss.item(), spotting_predictions.shape[0]*spotting_predictions.shape[1])
        accuracy_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        precision_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        recall_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        f1_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        if umt5_enabled:
            meteor_metric.update(meteor_calculator.compute(predictions=caption_logits_to_string(caption_logits), references=batch['captions'])['meteor'])

        step_loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose or i+1 == len(data_loader):
            progress_bar.set_postfix(
                lr=scheduler.get_last_lr()[0],
                loss=loss_metric.avg,
                accuracy=accuracy_metric.compute().item(),
                precision=precision_metric.compute().item(),
                recall=recall_metric.compute().item(),
                f1=f1_metric.compute().item(),
                meteor=meteor_metric.avg,
            )

    average_loss = loss_metric.avg
    average_accuracy = accuracy_metric.compute()
    average_precision = precision_metric.compute()
    average_recall = recall_metric.compute()
    average_f1 = f1_metric.compute()
    average_meteor = meteor_metric.avg

    tb_writer.add_scalar('lr/train', scheduler.get_last_lr()[0], epoch_index)
    tb_writer.add_scalar('Loss/train', average_loss, epoch_index)
    tb_writer.add_scalar('accuracy/train', average_accuracy, epoch_index)
    tb_writer.add_scalar('precision/train', average_precision, epoch_index)
    tb_writer.add_scalar('recall/train', average_recall, epoch_index)
    tb_writer.add_scalar('f1/train', average_f1, epoch_index)
    tb_writer.add_scalar('meteor/train', average_meteor, epoch_index)

    loss_metric.reset()
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    meteor_metric.reset()

    return average_loss, average_accuracy, average_precision, average_recall, average_f1, average_meteor

def evaluate_one_epoch(model, data_loader, short_memory_len, umt5_enabled, exclude_bg, epoch_index, tb_writer, device, verbose=0):
    loss_metric = AverageMeter()
    accuracy_metric = BinaryAccuracy()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    f1_metric = BinaryF1Score()
    meteor_metric = AverageMeter()

    progress_bar = tqdm(enumerate(data_loader), desc=f'Evaluate epoch {epoch_index+1}', total=len(data_loader))
    for i, batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        spotting_logits, caption_logits, step_loss = model(**batch)
        spotting_predictions = torch.sigmoid(spotting_logits)

        loss_metric.update(step_loss.item(), spotting_predictions.shape[0]*spotting_predictions.shape[1])
        accuracy_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        precision_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        recall_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        f1_metric.update(spotting_predictions[:, 1, exclude_bg:].reshape(-1), batch['spotting_labels'][:, 1, exclude_bg:].reshape(-1).int())
        if umt5_enabled:
            meteor_metric.update(meteor_calculator.compute(predictions=caption_logits_to_string(caption_logits), references=batch['captions'])['meteor'])

        if verbose or i+1 == len(data_loader):
            progress_bar.set_postfix(
                loss=loss_metric.avg,
                accuracy=accuracy_metric.compute().item(),
                precision=precision_metric.compute().item(),
                recall=recall_metric.compute().item(),
                f1=f1_metric.compute().item(),
                meteor=meteor_metric.avg,
            )

    average_loss = loss_metric.avg
    average_accuracy = accuracy_metric.compute()
    average_precision = precision_metric.compute()
    average_recall = recall_metric.compute()
    average_f1 = f1_metric.compute()
    average_meteor = meteor_metric.avg

    tb_writer.add_scalar(f'Loss/valid', average_loss, epoch_index)
    tb_writer.add_scalar(f'accuracy/valid', average_accuracy, epoch_index)
    tb_writer.add_scalar(f'precision/valid', average_precision, epoch_index)
    tb_writer.add_scalar(f'recall/valid', average_recall, epoch_index)
    tb_writer.add_scalar(f'f1/valid', average_f1, epoch_index)
    tb_writer.add_scalar(f'meteor/valid', average_meteor, epoch_index)

    loss_metric.reset()
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    meteor_metric.reset()

    return average_loss, average_accuracy, average_precision, average_recall, average_f1, average_meteor

def evaluate_one_epoch_sn(model, dataloader, dataset, split, umt5_enabled, epoch_index, base_dir, states_dir, tb_writer, device):
    # loss_metric = AverageMeter()
    model.train(False)
    predictions = {}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc=f"Predicting", total=len(dataloader)):
            batch = move_batch_to_device(batch, device)
            spotting_logits, caption_logits, _ = model(**batch, num_beams=1, max_new_tokens=50)
            spotting_predictions = torch.sigmoid(spotting_logits)
            if caption_logits is not None:
                caption_predictions = caption_ids_to_string(caption_logits)
            
            # loss_metric.update(step_loss.item(), spotting_predictions.shape[0]*spotting_predictions.shape[1])
            for sample_index in range(spotting_predictions.shape[0]):
                video_index, start_frame_index = batch['indices'][sample_index]
                game_path = dataset.get_game_path(video_index)
                if not game_path in predictions:
                    predictions[game_path] = {
                        'UrlLocal': game_path,
                        'predictions': []
                    }
                middle_point = start_frame_index+1.5*dataset.short_memory_len
                position = dataset.frame_index_to_position(middle_point)
                if caption_logits is not None:
                    events_caption_predictions = caption_predictions[sample_index].split('@')
                    event_index = 0
                for label_value, label_index in dataset.labels_dict.items():
                    if spotting_predictions[sample_index][1][label_index].item() >= 0.5:
                        minute = (position // 1000) // 60
                        second = (position // 1000) % 60
                        half = "1" if "1_224p_h264" in dataset.get_frames_path(video_index) else "2"
                        # if events_caption_predictions[event_index].strip() != '':
                        predictions[game_path]['predictions'].append(
                            {
                                "gameTime": f"{half} - {str(minute).rjust(2, '0')}:{str(second).rjust(2, '0')}",
                                "label": label_value,
                                "position": position,
                                "half": half,
                                "confidence": spotting_predictions[sample_index][1][label_index].item(),
                                # 'comment': events_caption_predictions[event_index]
                                'comment': events_caption_predictions[event_index] if caption_logits is not None and event_index < len(events_caption_predictions) else ''
                            }
                        )
                        if caption_logits is not None:
                            event_index += 1
                        # if event_index >= len(events_caption_predictions):
                        #     break

    for game_path, predictions_content in tqdm(predictions.items(), desc=f"Writing predictions"):
        output_dir = f"{states_dir}/outputs_temp/{game_path}/"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/results_dense_captioning.json", "w") as result_file:
            json.dump(predictions_content, result_file)

    results = sn_evaluate(
        SoccerNet_path=f"{base_dir}/data/",
        Predictions_path=f"{states_dir}/outputs_temp/",
        split=split,
        version=2,
        prediction_file="results_dense_captioning.json",
        include_SODA=False
    )

    # tb_writer.add_scalar(f'Loss/valid sn', loss_metric.avg, epoch_index)
    if umt5_enabled:
        tb_writer.add_scalar(f'Bleu_1/valid sn', results['Bleu_1'], epoch_index)
        tb_writer.add_scalar(f'Bleu_2/valid sn', results['Bleu_2'], epoch_index)
        tb_writer.add_scalar(f'Bleu_3/valid sn', results['Bleu_3'], epoch_index)
        tb_writer.add_scalar(f'Bleu_4/valid sn', results['Bleu_4'], epoch_index)
        tb_writer.add_scalar(f'meteor/valid sn', results['METEOR'], epoch_index)
        tb_writer.add_scalar(f'ROUGE_L/valid sn', results['ROUGE_L'], epoch_index)
        tb_writer.add_scalar(f'CIDEr/valid sn', results['CIDEr'], epoch_index)
    tb_writer.add_scalar(f'recall/valid sn', results['Recall'], epoch_index)
    tb_writer.add_scalar(f'precision/valid sn', results['Precision'], epoch_index)
    tb_writer.add_scalar(f'f1/valid sn', (2*results['Precision']*results['Recall'])/(results['Precision']+results['Recall']), epoch_index)
    
    shutil.rmtree(f"{states_dir}/outputs_temp/")
    return results['METEOR']
