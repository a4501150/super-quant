| config | phase | prompt | cached | from disk | TTFT s (each session) | decode tok/s |
|---|---|---|---|---|---|---|
| ram | cold | 20835 | 0/0/0 | n/n/n | 1.700/1.700/1.700 | 294/311/319 |
| ram | repeat | 20835 | 20828/20801/20851 | n/n/n | 0.020/0.021/0.020 | 321/308/320 |
| ram | after restart | 20835 | 0/0/0 | n/n/n | 1.800/1.700/1.700 | 294/311/312 |
| ram+disk | cold | 20790 | 0/0/0 | n/n/n | 1.800/1.700/1.700 | 328/309/335 |
| ram+disk | repeat | 20790 | 20783/20815/20843 | n/n/n | 0.024/0.020/0.021 | 352/306/334 |
| ram+disk | after restart | 20790 | 20783/20815/20843 | y/y/y | 0.060/0.054/0.054 | 318/307/335 |
| disk | cold | 20828 | 0/0/0 | n/n/n | 1.800/1.800/1.800 | 236/344/225 |
| disk | repeat | 20828 | 20821/20799/20839 | y/y/y | 0.081/0.057/0.074 | 250/339/223 |
| disk | after restart | 20828 | 20821/20799/20839 | y/y/y | 0.078/0.074/0.072 | 237/339/223 |

| config | Disk writes during boot 1 | Disk used at boot 2 | stop s | tiers v/r/s | VRAM GiB (experts / KV / other) |
|---|---|---|---|---|---|
| ram | 0.0 MiB (0 images, 0 blocks) | 0.0 MiB | 1.0 | 24576/0/0 | 63.31 / 19.27 / 7.67 |
| ram+disk | 1166.0 MiB (6 images, 321 blocks) | 2596.7 MiB | 1.4 | 24576/0/0 | 63.31 / 19.27 / 7.67 |
| disk | 1657.7 MiB (8 images, 559 blocks) | 2144.5 MiB | 0.8 | 24576/0/0 | 63.31 / 19.27 / 8.02 |
