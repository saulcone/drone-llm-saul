import os
import shutil

# Create backup directory
backup_dir = r"C:\Users\saulc\Desktop\Cuarto\TFG\drone-llm-saul\prompts\backup_fsm"
os.makedirs(backup_dir, exist_ok=True)

# Move state_*.txt files to backup
prompts_dir = r"C:\Users\saulc\Desktop\Cuarto\TFG\drone-llm-saul\prompts"
moved_files = []

for file in os.listdir(prompts_dir):
    if file.startswith("state_") and file.endswith(".txt"):
        src = os.path.join(prompts_dir, file)
        dst = os.path.join(backup_dir, file)
        shutil.move(src, dst)
        moved_files.append(file)

print(f"Backup completed! Moved {len(moved_files)} files:")
for f in moved_files:
    print(f"  - {f}")
