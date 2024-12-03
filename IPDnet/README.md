* For Train,
  
  ```
  python runIPDnetOn.py fit --data.batch_size=[*,*] --trainer.devices=*,*
  ```
  
  * For test,
  
  ```
  python runIPDnetOn.py test  --ckpt_path logs/MyModel/version_x/checkpoints/**.ckpt --trainer.devices=*,*
  ```
