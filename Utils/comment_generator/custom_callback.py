# filepath: /home/sunesh/NLP/Multi_Label_Toxic_Comment_Classifier/Utils/comment_generator/custom_callback.py
from transformers import TrainerCallback
from datetime import datetime

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total steps planned: {state.max_steps}\n")
            f.write(f"{'='*60}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = datetime.now()
            elapsed = current_time - self.start_time if self.start_time else None
            train_loss = logs.get('train_loss', 'N/A')
            loss_str = f"{train_loss:.4f}" if isinstance(train_loss, float) else str(train_loss)
            lr = logs.get('learning_rate', 'N/A')
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
            with open(self.log_file_path, 'a') as f:
                f.write(f"Step {state.global_step}: Loss={loss_str}, LR={lr_str}, Elapsed={elapsed}\n")

    def on_save(self, args, state, control, **kwargs):
        with open(self.log_file_path, 'a') as f:
            f.write(f"Model checkpoint saved at step {state.global_step}\n")

    def on_train_end(self, args, state, control, **kwargs):
        end_time = datetime.now()
        total_time = end_time - self.start_time if self.start_time else None
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_time}\n")
            f.write(f"Final step: {state.global_step}\n")
            f.write(f"{'='*60}\n\n")