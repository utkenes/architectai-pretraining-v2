# ArchitectAI Stage 5 Colab execution

1. Enable a GPU runtime and clone this repository.
2. Check out the dataset manifest's exact `build_git_sha`.
3. Install dependencies: `pip install -e ".[training,benchmark]"`.
4. Mount Google Drive and copy `architectai_dapt_dataset_v2.zip` from Drive.
5. Verify before model loading:
   `architectai-pretraining dapt verify-data --manifest /content/architectai_dapt_dataset_v2.zip`.
6. Inspect hardware: `architectai-pretraining dapt preflight`.
7. Run the frozen real baseline with Qwen3-8B, 4-bit inference, batch size 1:
   `architectai-pretraining benchmark baseline --quantization 4bit --device cuda`.
8. Save and inspect the baseline report; do not modify benchmark v1.
9. Explicitly select `qlora`, `lora`, or `full_parameter` in `configs/dapt.yaml` after reviewing preflight.
10. Run 10–20 smoke steps with a Drive-backed output directory:
    `architectai-pretraining dapt smoke --config configs/dapt.yaml --output-dir /content/drive/MyDrive/ArchitectAI/checkpoints/stage5a`.
11. Resume from the saved checkpoint:
    `architectai-pretraining dapt smoke --config configs/dapt.yaml --resume-from /content/drive/MyDrive/ArchitectAI/checkpoints/stage5a/checkpoint-10`.
12. Make the GO/NO-GO decision only after loss, evaluation, checkpoint, resume, and VRAM checks succeed.
