"""Temizleme, RFM, K-means segmentasyonu ve sonraki dönemle doğrulama.

Girdi : data/online_retail_II.csv     (01_prepare.py çıktısı)
Çıktı : data/rfm_segments.csv         (müşteri bazında RFM + segment + sonraki dönem)
        outputs/summary.json          (rapordaki tüm rakamlar)
        outputs/*.png                 (2 grafik)

Tasarım: segmentler ilk yılın verisinden (01.12.2009 – 30.11.2010) kurulur. İkinci yıl
(01.12.2010 – 09.12.2011) hiç kullanılmaz ve yalnızca "segmentler ertesi yılki davranışı
ayırıyor mu?" sorusu için saklanır. Bir tam yıl olduğu için mevsim etkisi karşılaştırmayı bozmaz.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = "data/online_retail_II.csv"
CUTOFF = pd.Timestamp("2010-12-01")  # bu tarihten itibaren doğrulama dönemi (ikinci yıl)
K = 4
SEED = 0
OUT = Path("outputs")

# ---------------------------------------------------------------- temizlik


def load_clean(path: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """Ham işlemleri temizler; her adımda kaç satır kaldığını ve müşterisiz satırların payını da döndürür."""
    df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    steps = [{"adim": "Ham veri", "satir": len(df)}]
    raw_revenue = df["Quantity"] * df["UnitPrice"]
    missing = df["CustomerID"].isna()
    stats = {"musterisiz_satir_payi_yuzde": round(float(missing.mean() * 100), 1),
             "musterisiz_ciro_payi_yuzde": round(float(raw_revenue[missing].sum() / raw_revenue.sum() * 100), 1)}

    def keep(mask_or_df, name):
        nonlocal df
        df = mask_or_df if isinstance(mask_or_df, pd.DataFrame) else df[mask_or_df]
        steps.append({"adim": name, "satir": len(df)})

    # İki Excel sayfası 1-9 Aralık 2010'da çakışır; birebir aynı satırların çoğu buradan gelir.
    keep(df.drop_duplicates(), "Birebir tekrar eden satırlar atıldı")
    keep(df["CustomerID"].notna(), "Müşteri numarası olmayanlar atıldı")
    # Ürün kodları beş rakamla başlar; POST, M, D, BANK CHARGES gibi kodlar kargo, düzeltme ve masraftır.
    keep(df["StockCode"].str.match(r"^\d{5}"), "Ürün olmayan satırlar atıldı (kargo, düzeltme, masraf)")
    keep(df["UnitPrice"] > 0, "Fiyatı sıfır olanlar atıldı")

    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    # İadeler atılmaz: ciroyu eksi yönde düzeltsinler diye tutulur (fatura no 'C' ile başlar).
    df["is_return"] = df["InvoiceNo"].str.startswith("C")
    return df, steps, stats


# ---------------------------------------------------------------- RFM


def build_rfm(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """R: son alışverişten bu yana gün, F: fatura sayısı, M: iadeler düşülmüş net ciro."""
    buys = tx[~tx["is_return"]]
    rfm = buys.groupby("CustomerID").agg(
        last=("InvoiceDate", "max"), first=("InvoiceDate", "min"), frequency=("InvoiceNo", "nunique"))
    rfm["monetary"] = tx.groupby("CustomerID")["revenue"].sum()
    rfm["recency"] = (as_of - rfm["last"]).dt.days
    rfm["tenure"] = (as_of - rfm["first"]).dt.days
    # Net cirosu sıfır ya da eksi olanlar (aldığının tamamını iade edenler) segmentlenmez.
    return rfm[rfm["monetary"] > 0].drop(columns=["last", "first"])


def features(rfm: pd.DataFrame) -> np.ndarray:
    """Üç değişken de sağa çarpık; log alıp standartlaştırmadan K-means en büyük birkaç müşteriye kilitlenir."""
    return StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))


def choose_k(x: np.ndarray) -> list[dict]:
    """Her K için siluet skoru ve kararlılık (yeniden örneklenen veriyle kurulan modelle uyum, ARI)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(2, 8):
        base = KMeans(k, n_init=10, random_state=SEED).fit(x)
        aris = []
        for s in range(10):
            sample = rng.choice(len(x), len(x), replace=True)
            boot = KMeans(k, n_init=5, random_state=s).fit(x[sample])
            aris.append(adjusted_rand_score(base.labels_, boot.predict(x)))
        rows.append({"k": k, "siluet": round(float(silhouette_score(x, base.labels_)), 3),
                     "kararlilik_ari": round(float(np.mean(aris)), 3)})
    return rows


def name_segments(rfm: pd.DataFrame) -> dict[int, str]:
    """Küme numaraları her çalıştırmada değişebilir; isimler profilden türetilir."""
    med = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].median()
    names = {med["monetary"].idxmax(): "Şampiyonlar", med["recency"].idxmax(): "Uykudakiler"}
    rest = med.drop(index=list(names)).sort_values("monetary")
    names[rest.index[0]] = "Yeni ve küçük"
    names[rest.index[1]] = "Düzenli"
    return names


# ---------------------------------------------------------------- grafikler

SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
BLUE, ORANGE = "#3987e5", "#d95926"
ORDER = ["Şampiyonlar", "Düzenli", "Yeni ve küçük", "Uykudakiler"]


def new_ax(title, subtitle, size=(9, 4.6)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#383835")
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(lambda v, _: f"%{v:.0f}")
    fig.text(0.02, 0.96, title, color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.885, subtitle, color=INK2, fontsize=10.5, va="top")
    return fig, ax


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def label(ax, bars):
    for b in bars:
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"%{b.get_width():.0f}",
                color=INK, fontsize=10, va="center")


def chart_shares(prof: pd.DataFrame, path: Path):
    top = prof.loc["Şampiyonlar"]
    fig, ax = new_ax(f"Şampiyonlar: müşteri payı %{top['musteri_payi']:.0f}, ciro payı %{top['ciro_payi']:.0f}",
                     "Segmentlerin ilk yıldaki müşteri ve ciro payı")
    y = np.arange(len(ORDER))[::-1]
    a = ax.barh(y + 0.19, prof.loc[ORDER, "musteri_payi"], height=0.36, color=BLUE,
                edgecolor=SURFACE, linewidth=2, label="Müşteri payı")
    b = ax.barh(y - 0.19, prof.loc[ORDER, "ciro_payi"], height=0.36, color=ORANGE,
                edgecolor=SURFACE, linewidth=2, label="Ciro payı")
    label(ax, a), label(ax, b)
    ax.set_yticks(y, ORDER)
    ax.set_xlim(0, max(prof["ciro_payi"].max(), prof["musteri_payi"].max()) * 1.15)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2, fontsize=10.5)
    save(fig, path)


def chart_holdout(prof: pd.DataFrame, path: Path):
    fig, ax = new_ax("Segmentler ertesi yılı ayırıyor", "Ertesi yıl yeniden alışveriş yapanların oranı",
                     size=(9, 3.8))
    y = np.arange(len(ORDER))[::-1]
    label(ax, ax.barh(y, prof.loc[ORDER, "sonraki_donem_alisveris_orani"], height=0.5, color=BLUE))
    ax.set_yticks(y, ORDER)
    ax.set_xlim(0, 100)
    save(fig, path)


# ---------------------------------------------------------------- akış


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    tx, steps, stats = load_clean(DATA)
    cal, hold = tx[tx["InvoiceDate"] < CUTOFF], tx[tx["InvoiceDate"] >= CUTOFF]

    rfm = build_rfm(cal, CUTOFF)
    x = features(rfm)
    k_table = choose_k(x)
    rfm["cluster"] = KMeans(K, n_init=10, random_state=SEED).fit_predict(x)
    rfm["segment"] = rfm["cluster"].map(name_segments(rfm))

    # İkinci yıl: aynı müşteriler ne yaptı?
    hold_buys = hold[~hold["is_return"]].groupby("CustomerID")["InvoiceNo"].nunique()
    rfm["next_orders"] = hold_buys.reindex(rfm.index).fillna(0).astype(int)
    rfm["next_revenue"] = hold.groupby("CustomerID")["revenue"].sum().reindex(rfm.index).fillna(0)
    rfm["next_bought"] = rfm["next_orders"] > 0
    rfm.drop(columns="cluster").to_csv("data/rfm_segments.csv", encoding="utf-8-sig")

    g = rfm.groupby("segment")
    prof = pd.DataFrame({
        "musteri_sayisi": g.size(),
        "musteri_payi": g.size() / len(rfm) * 100,
        "ciro_payi": g["monetary"].sum() / rfm["monetary"].sum() * 100,
        "medyan_son_alisveris_gun": g["recency"].median(),
        "medyan_siparis": g["frequency"].median(),
        "medyan_ciro": g["monetary"].median(),
        "medyan_musterilik_gun": g["tenure"].median(),
        "sonraki_donem_alisveris_orani": g["next_bought"].mean() * 100,
        "sonraki_donem_ortalama_ciro": g["next_revenue"].mean(),
        "sonraki_donem_ciro_payi": g["next_revenue"].sum() / rfm["next_revenue"].sum() * 100,
    }).loc[ORDER]

    chart_shares(prof, OUT / "1_musteri_ciro_payi.png")
    chart_holdout(prof, OUT / "2_sonraki_donem.png")

    summary = {
        "veri": {
            "donem": f"{tx['InvoiceDate'].min():%d.%m.%Y} – {tx['InvoiceDate'].max():%d.%m.%Y}",
            "ham_satir": steps[0]["satir"],
            "temizlik_adimlari": steps,
            **stats,
            "musteri_sayisi": int(tx["CustomerID"].nunique()),
            "iade_satiri": int(tx["is_return"].sum()),
            "iade_tutari": round(float(-tx.loc[tx["is_return"], "revenue"].sum())),
            "brut_ciro": round(float(tx.loc[~tx["is_return"], "revenue"].sum())),
        },
        "tasarim": {
            "segment_donemi": f"{tx['InvoiceDate'].min():%d.%m.%Y} – {CUTOFF - pd.Timedelta(days=1):%d.%m.%Y}",
            "dogrulama_donemi": f"{CUTOFF:%d.%m.%Y} – {tx['InvoiceDate'].max():%d.%m.%Y}",
            "dogrulama_donemi_yeni_musteri": int(len(set(hold.loc[~hold["is_return"], "CustomerID"]) - set(rfm.index))),
            "segmentlenen_musteri": int(len(rfm)),
            "tek_siparisli_musteri_payi_yuzde": round(float((rfm["frequency"] == 1).mean() * 100), 1),
        },
        "k_secimi": k_table,
        "secilen_k": K,
        "segmentler": prof.round(1).to_dict(orient="index"),
        "sonraki_donem_genel_alisveris_orani": round(float(rfm["next_bought"].mean() * 100), 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
