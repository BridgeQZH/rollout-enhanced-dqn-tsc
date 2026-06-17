# media/

Curated assets referenced by the root `README.md`.

- `final_comparison.mp4` — the D=2000 demo (SUMO-GUI + live telemetry, DQN vs
  Rollout-DQN), **compressed to ~36 MB** (1280-wide, H.264) so it stays under the
  50 MB GitHub budget and embeds natively in the README.

The full-resolution master (1728×1080, ~93 MB) lives in `deck/videos/` and is
git-ignored. To re-derive this clip from a fresh recording:

```bash
ffmpeg -y -i deck/videos/final_comparison.mp4 -c:v libx264 -b:v 3200k -pass 1 -an -vf scale=1280:-2 -preset medium -f mp4 /dev/null
ffmpeg -y -i deck/videos/final_comparison.mp4 -c:v libx264 -b:v 3200k -pass 2 -an -vf scale=1280:-2 -preset medium -pix_fmt yuv420p -movflags +faststart media/final_comparison.mp4
```

> GitHub note: a relative `<video src=...>` in the README plays in most modern
> browsers and on GitHub when the file is committed (<100 MB hard limit, 50 MB
> soft warning). If inline playback is ever blocked, the README also links the
> file directly, and you can attach it to a Release for a guaranteed CDN URL.
