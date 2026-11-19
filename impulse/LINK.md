# Impulse talk

**Topic:** Latent Space: *How does AI know two buildings look alike? When buildings become numbers* (understanding latent space through architectural visual search)
**Day:** Tuesday 17 November 2026
**Slides (Google Slides, anyone with the link can view):** https://docs.google.com/presentation/d/109lvBtwZO-syQzukYkceIbkNtaKl2TnJE4GAtRkwQx4/view
**Slides (PDF, in this repository):** [latent_space_impulse_slides.pdf](latent_space_impulse_slides.pdf), 5 slides

| # | Slide | What it shows |
|---|---|---|
| 1 | The question | How does AI know two buildings look alike? |
| 2 | The use case | Google Lens finds visual matches for a photo of Torre Glòries (Barcelona) |
| 3 | How it works inside | System diagram: browser → encoder → embedding → vector index (reference images encoded in advance) → image results |
| 4 | What it means for us | Looking alike doesn't mean performing alike. Our RIAWELC autoencoder also learns a latent code, but flags weld anomalies by reconstruction error, not by similarity search |
| 5 | Closing question | What does similar really mean: shape, material or function? |

Image credits are on the slides: photo by Diliff, Wikimedia Commons, CC BY-SA 3.0; Google Lens screenshot, 16 November 2026. Slide 3 is a conceptual architecture, not Google's internal system (references: CLIP, Radford et al. 2021; FAISS, Johnson et al. 2017).
