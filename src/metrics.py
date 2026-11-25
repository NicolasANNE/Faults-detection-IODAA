from scipy import ndimage as ndi
import numpy as np

R_PX, ALPHA, BETA = 3, 0.2, 0.8
OFFSETS = [(dy, dx, 1 - np.hypot(dy, dx) / R_PX)
           for dy in range(-R_PX, R_PX + 1) for dx in range(-R_PX, R_PX + 1) if np.hypot(dy, dx) < R_PX]


class DTI:
    """Distance-weighted Tversky index sur une zone `region` (la vérité et les prédictions hors zone sont ignorées)."""

    def __init__(self, y, region, alpha=ALPHA, beta=BETA):
        """
        y: vérité terrain (1 pour les pixels de la faille, 0 sinon)
        region: zone d'intérêt (True pour les pixels à considérer)
        alpha, beta: paramètres du Tversky index
        """
        self.alpha, self.beta, self.region = alpha, beta, region
        G = (y == 1) & region # pixels de la faille dans la zone d'intérêt
        self.gy, self.gx = np.nonzero(G)
        d = ndi.distance_transform_edt(~G)
        d = np.asarray(d, np.float32)
        self.w_fp = (1 - np.clip(1 - d / R_PX, 0, None)).astype(np.float32)   # 1 - max_g k(d(x,g))
        self.n_fault = len(self.gy)

    def parts(self, p):
        p = np.where(self.region, np.nan_to_num(p), 0).astype(np.float32)
        pp = np.pad(p, R_PX)
        m = np.zeros(self.n_fault, np.float32)
        for dy, dx, k in OFFSETS:
            m = np.maximum(m, pp[self.gy + R_PX + dy, self.gx + R_PX + dx] * k)
        tp = float(m.sum(dtype=np.float64))
        fn = self.n_fault - tp
        fp = float((p * self.w_fp).sum(dtype=np.float64))
        return tp, fp, fn

    def __call__(self, p):
        tp, fp, fn = self.parts(p)
        return tp / (tp + self.alpha * fp + self.beta * fn + 1e-9)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # vérification sur un cas jouet : une faille verticale, prédiction parfaite puis décalée d'un pixel
    y_t = np.zeros((20, 20), np.int8); y_t[:, 10] = 1
    ev = DTI(y_t, np.ones_like(y_t, bool))
    p_ok = y_t.astype(np.float32)
    p_shift = np.roll(p_ok, 1, axis=1)
    print("prédiction parfaite  :", round(ev(p_ok), 3))
    print("décalée de 1 pixel   :", round(ev(p_shift), 3), "(attendu : TP=2/3 par pixel, FN=1/3, FP=1/3 -> 0.667)")
    print("rien prédit          :", round(ev(np.zeros_like(p_ok)), 3))
    print("tout à 1 (20x20)     :", round(ev(np.ones_like(p_ok)), 3))
    # plot des exemples et des prédictions et des scores
    plt.figure(figsize=(8, 2))
    scale = 5
    y_plot = np.repeat(np.repeat(y_t, scale, axis=0), scale, axis=1)
    p_ok_plot = np.repeat(np.repeat(p_ok, scale, axis=0), scale, axis=1)
    p_shift_plot = np.repeat(np.repeat(p_shift, scale, axis=0), scale, axis=1)

    for i, (title, prediction) in enumerate(
        ((f"Vérité: {ev(p_ok):.3f}", y_plot), (f"Prédiction parfaite: {ev(p_ok):.3f}", p_ok_plot),
         (f"Prédiction décalée: {ev(p_shift):.3f}", p_shift_plot)), 1
    ):
        ax = plt.subplot(1, 3, i)
        ax.imshow(prediction, cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.contour(y_plot, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(title)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
