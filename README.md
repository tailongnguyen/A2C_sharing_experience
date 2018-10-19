# A2C_sharing_experience
Multi-task learning with Advantage Actor Critic  and sharing experience 

## To run the code

I implemented 3 versions: multithread (`train.py`), multiprocess (`train_multiprocess.py`) and multiprocess + ppo (`train_ppo.py`)

## Noticable changes

Reduce state space by removing the redundant states. Policy network and Value network can be joint or disjoint as specified in arguments. 

## Notes on results

- Network của anh Long hội tụ kém do để sai hàm value loss
- Khi chạy code multithread (thường để num_episodes là 10 và numsteps là 50) thì cái share không ngon được bằng cái none (cái none hội tụ quá ngon), vừa hội tụ sau lại còn rewards thấp hơn. Xem ở `logs/2018-10-15_23-28-26_plot_adv` và `logs/2018-10-16_15-32-04_plot_adv_15` (new_iw là kiểu importance weights hiện tại, weird là nó không tốt bằng kiểu cũ)

- Khi chạy code multiprocess (thường để num_episodes là 10 và numsteps là 15) thì cái share hội tụ trước nhưng rewards lại thấp hơn. 
  -  Ý tưởng cải tiến: share decay!!, tạo cơ chế để chỉ tập trung share những epoch đầu, `share_choice = np.random.choice([1, 0], p = [share_decay ** epoch, 1 - share_decay ** epoch])`. Kết quả xem ở `logs/2018-10-19_12-05-37_multiprocess_test_noiw` (share_decay là 0.989). Ngoài ra cũng thử sharecut, tức là qua một epoch nào đấy là không share nữa, kết quả ở `logs/2018-10-17_21-54-23_multiprocess_none_vs_share_vs_sharecut`
  -  Thay đổi importance weight: chưa nghĩ ra
-  Cải thiện chất lượng hội tụ của bằng PPO. Khi gắn PPO vào code multiprocess thì results khá là promising, share hội tụ có tốt hơn none tuy rằng độ chênh lệch không lớn. Xem ở logs/2018-10-19_15-03-32_ppo_share_vs_none



