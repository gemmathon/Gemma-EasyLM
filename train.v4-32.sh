export TPU_NAME='tpu-32'
export TPU_USER='pj2417'
export ZONE='us-central2-b'

echo "[local] Killing TPU"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "killall screen"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "rm screenlog.0"

echo "[local] Git pull"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone $ZONE --worker=all --command \
"cd Gemma-EasyLM && git fetch origin && \
git reset --hard origin/main && rm /home/$TPU_USER/Gemma-EasyLM/train.sh"

echo "[local] Set runner.sh"
# Log per 128 * 50 steps, matching the gradient accumulation steps = Real 1 step
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone $ZONE --worker=all --command "
cat > /home/$TPU_USER/Gemma-EasyLM/runner.sh << 'EOF'
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python -m EasyLM.models.gemma.gemma_train \
--load_checkpoint=trainstate_params::/home/$TPU_USER/streaming_train_state \
--mesh_dim=1,-1,4 \
--dtype=bf16 \
--total_steps=1600000 \
--log_freq=64 \
--save_model_freq=160000 \
--save_milestone_freq=160000 \
--train_dataset.type='json' \
--train_dataset.text_processor.fields='text' \
--train_dataset.json_dataset.path='gs://gemma_train/data.jsonl' \
--train_dataset.json_dataset.seq_length=8192 \
--train_dataset.json_dataset.batch_size=4 \
--optimizer.accumulate_gradient_steps=256 \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=0.00005 \
--optimizer.adamw_optimizer.end_lr=0.000001 \
--optimizer.adamw_optimizer.lr_warmup_steps=86000 \
--optimizer.adamw_optimizer.lr_decay_steps=1600000 \
--checkpointer.save_optimizer_state=True \
--checkpointer.float_dtype=bf16 \
--logger.online=True \
--logger.output_dir=gs://gemma-train/gemma-checkpoint \
--logger.project='gemma-ko-2b-dev'
EOF
chmod +x /home/$TPU_USER/Gemma-EasyLM/runner.sh"

echo "[local] RUN!!!"

gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone us-central2-b --worker=all --command \
"screen -L -d -m bash -i -c 'export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400; \
cd Gemma-EasyLM; /home/$TPU_USER/Gemma-EasyLM/runner.sh'"
