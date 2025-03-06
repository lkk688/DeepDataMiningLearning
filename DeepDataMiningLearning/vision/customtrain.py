import json
import os
import math
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepDataMiningLearning.vision.myevaluate import evaluate_dataset

def custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator=None, do_evaluate=False, logger=None):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Get the metric function
    #metric = evaluate.load("accuracy") #replaced with metriceval

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if logger:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                if args.task == "object-detection":
                    # pixel_values = batch["pixel_values"].to(device)
                    # pixel_mask = batch["pixel_mask"].to(device)
                    # labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                    pixel_values = batch["pixel_values"]
                    pixel_mask = batch["pixel_mask"]
                    labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                    loss_dict = outputs.loss_dict
                    #print(loss_dict)
                else:
                    outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        if do_evaluate:
            eval_metric = evaluate_dataset(model, eval_dataloader, args.task, metriceval, device, image_processor=image_processor, accelerator=accelerator)
    
            # model.eval()
            # for step, batch in enumerate(eval_dataloader):
            #     with torch.no_grad():
            #         outputs = model(**batch)
            #     predictions = outputs.logits.argmax(dim=-1)
            #     predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            #     metriceval.add_batch(
            #         predictions=predictions,
            #         references=references,
            #     )

            # eval_metric = metriceval.compute()#metric.compute()
            if logger:
                logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        #saving the models
        if args.checkpointing_steps == "epoch" and epoch % args.saving_everynsteps ==0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            if do_evaluate:
                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)
        #push to hub
        if args.hubname:
            unwrapped_model.push_to_hub(args.hubname)
            image_processor.push_to_hub(args.hubname)
            
class ImageDistilTrainer:
    def __init__(self, teacher_model, student_model, temperature, lambda_param, optimizer, device=None):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.optimizer = optimizer
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.teacher.to(self.device)
        self.teacher.eval()
        self.student.to(self.device)

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        student_output = self.student(inputs)
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        # Compute soft targets
        soft_teacher = F.softmax(teacher_output / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = F.cross_entropy(student_output, labels)  # Example: Assuming cross-entropy for true label loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                loss = self.train_step(inputs, labels)
                epoch_loss += loss
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, Loss: {loss:.4f}")
            print(f"Epoch {epoch+1} average loss: {epoch_loss / len(train_loader):.4f}")

# Example usage
# Assuming you have your teacher_model, student_model, train_loader, optimizer, etc.
# trainer = ImageDistilTrainer(teacher_model, student_model, temperature=2.0, lambda_param=0.5, optimizer=optimizer)
# trainer.train(train_loader, epochs=10)