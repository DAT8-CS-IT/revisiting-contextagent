### Tool-calling F1 per scenario

| Configuration | runs | Chitchat (n=54) | Cooking (n=36) | Health (n=42) | Leisure (n=54) | Outdoors (n=54) | Shopping (n=31) | Travel (n=37) | Work (n=49) | Others (n=43) | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama dense (r=8, 5e-5) | 3 | 0.622 | 0.738 | 0.583 | 0.852 | 0.807 | 0.651 | 0.759 | 0.863 | 0.728 | 0.740 |
| Llama dense, seed 42 | 1 | 0.555 | 0.806 | 0.596 | 0.839 | 0.793 | 0.647 | 0.778 | 0.863 | 0.714 | 0.735 |
| Llama combined 16-bit | 3 | 0.678 | 0.809 | 0.616 | 0.869 | 0.841 | 0.708 | 0.795 | 0.856 | 0.721 | 0.771 |
| Llama combined 4-bit | 3 | 0.692 | 0.786 | 0.648 | 0.908 | 0.862 | 0.724 | 0.768 | 0.884 | 0.745 | 0.787 |
| Wanda 50% (re-masked) | 3 | 0.423 | 0.633 | 0.526 | 0.805 | 0.748 | 0.478 | 0.632 | 0.763 | 0.613 | 0.634 |
| Wanda 50% (mask-aware recovery) | 1 | 0.591 | 0.713 | 0.556 | 0.842 | 0.810 | 0.512 | 0.702 | 0.827 | 0.692 | 0.706 |
| Qwen-7B combined 16-bit (r=32, 1e-4) | 3 | 0.722 | 0.823 | 0.558 | 0.852 | 0.781 | 0.685 | 0.713 | 0.843 | 0.750 | 0.753 |
| Qwen-1.5B (r=32, 1e-4) | 1 | 0.624 | 0.826 | 0.610 | 0.786 | 0.811 | 0.591 | 0.636 | 0.785 | 0.631 | 0.707 |

### Effect per scenario (F1 difference, 95% bootstrap CI) and heterogeneity test

- 4-bit vs 16-bit (Llama, combined): overall +0.016; heterogeneity p = 0.790
  Chitchat +0.014 [-0.05,+0.08], Cooking -0.023 [-0.09,+0.05], Health +0.032 [-0.03,+0.10], Leisure +0.039 [+0.00,+0.08], Outdoors +0.021 [-0.01,+0.06], Shopping +0.016 [-0.04,+0.07], Travel -0.027 [-0.09,+0.03], Work +0.028 [-0.03,+0.09], Others +0.024 [-0.03,+0.08]
- Wanda 50% re-masked vs dense (Llama): overall -0.101; heterogeneity p = 0.335
  Chitchat -0.132 [-0.21,-0.05], Cooking -0.173 [-0.29,-0.05], Health -0.071 [-0.19,+0.06], Leisure -0.034 [-0.09,+0.02], Outdoors -0.045 [-0.13,+0.05], Shopping -0.169 [-0.31,-0.03], Travel -0.146 [-0.25,-0.04], Work -0.100 [-0.17,-0.03], Others -0.101 [-0.18,-0.03]
- Wanda 50% mask-aware vs dense (Llama): overall -0.029; heterogeneity p = 0.436
  Chitchat +0.036 [-0.06,+0.13], Cooking -0.093 [-0.21,+0.02], Health -0.040 [-0.17,+0.09], Leisure +0.003 [-0.07,+0.06], Outdoors +0.017 [-0.09,+0.13], Shopping -0.135 [-0.28,+0.01], Travel -0.076 [-0.19,+0.04], Work -0.036 [-0.11,+0.04], Others -0.022 [-0.11,+0.06]
- Qwen-1.5B vs Qwen-7B (r=32, 1e-4): overall -0.047; heterogeneity p = 0.152
  Chitchat -0.098 [-0.19,-0.01], Cooking +0.003 [-0.10,+0.11], Health +0.052 [-0.05,+0.17], Leisure -0.066 [-0.13,-0.01], Outdoors +0.030 [-0.05,+0.12], Shopping -0.094 [-0.22,+0.03], Travel -0.077 [-0.21,+0.06], Work -0.058 [-0.15,+0.02], Others -0.119 [-0.22,-0.03]
