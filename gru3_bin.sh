Launching DEFOM-PIVNO bins training from scratch:
  GPUs=0, global_batch=2, local_batch=2
  data=/home/share/yijiayi/sceneflow
  steps=200000, all_parameter_lr=0.0002
  output=checkpoints/defom_pivno_gated_gru3_bins_scratch_d768_320x768_b2_200k
Traceback (most recent call last):
  File "/home/u2025140689/anaconda3/envs/defomstereo310/bin/torchrun", line 6, in <module>
    sys.exit(main())
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/run.py", line 892, in main
    run(args)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 261, in launch_agent
    result = agent.run()
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 138, in wrapper
    result = f(*args, **kwargs)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 711, in run
    result = self._invoke_run(role)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 864, in _invoke_run
    self._initialize_workers(self._worker_group)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 138, in wrapper
    result = f(*args, **kwargs)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 683, in _initialize_workers
    self._rendezvous(worker_group)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 138, in wrapper
    result = f(*args, **kwargs)
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 500, in _rendezvous
    rdzv_info = spec.rdzv_handler.next_rendezvous()
  File "/home/u2025140689/anaconda3/envs/defomstereo310/lib/python3.10/site-packages/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py", line 67, in next_rendezvous
    self._store = TCPStore(  # type: ignore[call-arg]
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29547, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use
ERROR conda.cli.main_run:execute(127): `conda run torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29547 train_stereo.py --distributed --launcher pytorch --gpu_ids 0 --batch_size 2 --num_workers 4 --train_datasets sceneflow --train_folds 1 --image_size 320 768 --max_disp 768 --mixed_precision --n_downsample 2 --n_gru_layers 3 --hidden_dims 128 128 128 --context_norm instance --train_iters 16 --valid_iters 32 --scale_iters 0 --corr_radius 4 --pivno_num_init_bins 48 --pivno_bins_max_offset 48.0 --pivno_init_corr_scale 10.0 --save_latest_ckpt_freq 1000 --save_ckpt_freq 20000 --val_freq 20000 --model defom_pivno_gated_gru3_bins --pivno_bins_stage joint --pivno_bins_pretrained_lr 0.0002 --name defom_pivno_gated_gru3_bins_scratch_d768_320x768_b2_200k --num_steps 200000 --lr 0.0002` failed. (See above for error)
