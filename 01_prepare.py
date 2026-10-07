"""Online Retail II Excel dosyasını tek bir CSV'ye çevirir -> data/online_retail_II.csv

Veri: UCI Machine Learning Repository, "Online Retail II" (Chen, 2019).
İngiltere merkezli bir online hediyelik eşya satıcısının 01.12.2009 – 09.12.2011 arası tüm işlemleri.
İndir: https://archive.ics.uci.edu/dataset/502/online+retail+ii  ->  data/online_retail_II.xlsx

Excel'de iki sayfa var (2009-2010 ve 2010-2011). 1-9 Aralık 2010 her iki sayfada da yer alır;
bu tekrarlar 02_segment.py içindeki temizlik adımında atılır.
"""
from pathlib import Path

import pandas as pd

SRC = Path("data/online_retail_II.xlsx")
OUT = Path("data/online_retail_II.csv")
EXPECTED_ROWS = 1_067_371


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} bulunamadı. Dosyayı UCI'dan indirip data/ klasörüne koy (adres dosyanın başında).")
    print("Excel okunuyor (bir iki dakika sürer)...")
    sheets = pd.read_excel(SRC, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(f"Beklenen {EXPECTED_ROWS} satır, bulunan {len(df)}. Dosya eksik ya da farklı sürüm.")
    df.to_csv(OUT, index=False)
    print(f"{len(df)} satır -> {OUT}")


if __name__ == "__main__":
    main()
