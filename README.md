# BGE-SigLIP: Unified Multimodal & Cross-Lingual Embeddings

Multimodal embedding model bridging SigLIP-2 and BGE-M3 into a unified vector space, with native cross-lingual image-text.
**BGE-SigLIP** bridges the gap between state-of-the-art vision and text embedding models. By fine-tuning **SigLIP-2**'s vision encoder directly into **BGE-M3**'s native 1024-dimensional vector space, this model enables seamless, high-performance multimodal retrieval without sacrificing cross-lingual capabilities.

## Features

* **Unified Vector Space:** Images and text are projected into the exact same 1024-d space.
* **Native Cross-Lingual Support:** Search images using queries in 100+ languages out-of-the-box.
* **Asymmetric Contrastive Fine-Tuning:** Preserving the rich textual semantic depth of BGE-M3.

