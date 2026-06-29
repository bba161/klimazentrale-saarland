"""
fetch_dwd.py – DWD Saarland Klimazentrale
=========================================
Läuft stündlich per GitHub Actions.
Lädt stündliche Temperaturdaten + Tagesniederschlag vom DWD,
berechnet Klimanormale und speichert alles als dwd_saarland.json.

Lokal ausführbar mit: python3 fetch_dwd.py
"""

import requests
import zipfile
import io
import json
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# ─── KONFIGURATION ───────────────────────────────────────────────
STATIONEN = {
    'ensheim':     {'name': 'Saarbrücken-Ensheim',     'id': '04336'},
    'burbach':     {'name': 'Saarbrücken-Burbach',      'id': '06217'},
    'berus':       {'name': 'Berus',                    'id': '00460'},
    'homburg':     {'name': 'Homburg',                  'id': '02331'},
    'merzig':      {'name': 'Merzig',                   'id': '03263'},
    'neunkirchen': {'name': 'Neunkirchen-Wellesweiler', 'id': '03545'},
    'nohfelden':   {'name': 'Nohfelden-Gonnesweiler',   'id': '03625'},
    'perl':        {'name': 'Perl-Nennig',              'id': '03904'},
    'schmelz':     {'name': 'Schmelz-Hüttersdorf',      'id': '04490'},
    'tholey':      {'name': 'Tholey',                   'id': '05029'},
    'weiskirchen': {'name': 'Weiskirchen',              'id': '05433'},
}

# Nur aktuelle Daten (recent) stündlich abrufen.
# Klimanormale (historical) werden nur neu berechnet wenn
# RECALC_HISTORICAL=true gesetzt ist (einmal täglich reicht).
RECALC_HISTORICAL = os.environ.get('RECALC_HISTORICAL', 'false').lower() == 'true'
OUTPUT_FILE = 'dwd_saarland.json'

BASE_TEMP = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/'
BASE_RAIN = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/'

# ─── HILFSFUNKTIONEN ─────────────────────────────────────────────
def get_zip_url(base_url, zeitraum, station_id):
    """Dateinamen der ZIP im DWD-Verzeichnis finden."""
    url = base_url + zeitraum + '/'
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a') if a.get('href', '').endswith('.zip')]
    treffer = [l for l in links if f'_{station_id}_' in l]
    return (url + treffer[0]) if treffer else None


def lade_zip(url):
    """ZIP herunterladen und Datendatei als DataFrame zurückgeben."""
    r = requests.get(url, timeout=120)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data_file = [f for f in z.namelist() if f.startswith('produkt_')][0]
        with z.open(data_file) as f:
            df = pd.read_csv(f, sep=';', encoding='latin-1')
    df.columns = df.columns.str.strip()
    return df.replace(-999.0, np.nan).replace(-999, np.nan)


def klimanormale_berechnen(df, wert_col, von_jahr, bis_jahr):
    """Klimanormale (Mittel + 10/90-Perzentile) je Kalendertag."""
    df = df.copy()
    df['tag'] = pd.to_datetime(df['tag'])
    mask = (df['tag'].dt.year >= von_jahr) & (df['tag'].dt.year <= bis_jahr)
    ref = df[mask].copy()
    ref['doy'] = ref['tag'].dt.dayofyear
    result = ref.groupby('doy')[wert_col].agg(
        mittel='mean',
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90)
    ).reset_index()
    for col in ['mittel', 'p10', 'p90']:
        result[col] = result[col].rolling(7, center=True, min_periods=1).mean().round(1)
    return {
        int(row['doy']): {
            'mittel': row['mittel'],
            'p10': row['p10'],
            'p90': row['p90']
        }
        for _, row in result.iterrows()
    }


def np_serial(obj):
    """NumPy-Typen für JSON serialisierbar machen."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f'Not JSON serializable: {type(obj)}')


# ─── SCHRITT 1: Bestehende JSON laden (falls vorhanden) ──────────
# Beim stündlichen Update überschreiben wir nur den "aktuell"-Block,
# die Klimanormale bleiben unverändert (spart Zeit und Bandbreite).
bestehend = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding='utf-8') as f:
        bestehend = json.load(f)
    print(f'Bestehende JSON geladen ({len(bestehend.get("klimanormale", {}))} Stationen mit Normalen)')
else:
    print('Keine bestehende JSON – vollständiger Erstlauf')
    RECALC_HISTORICAL = True


# ─── SCHRITT 2: Historische Daten + Klimanormale (bei Bedarf) ────
klimanormale = bestehend.get('klimanormale', {})

if RECALC_HISTORICAL:
    print('\n=== Berechne Klimanormale (historical) ===')
    alle_hist_temp = {}
    alle_hist_regen = {}

    for key, st in STATIONEN.items():
        sid = st['id']
        print(f"\n{st['name']} (ID {sid})")

        # Temperaturdaten historisch
        try:
            url = get_zip_url(BASE_TEMP, 'historical', sid)
            if url:
                df = lade_zip(url)
                df['datum'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H')
                df['tag'] = df['datum'].dt.date.astype(str)
                daily = df.groupby('tag')['TT_TU'].agg(tmax='max', tmean='mean').reset_index()
                daily['tmax'] = daily['tmax'].round(1)
                daily['tmean'] = daily['tmean'].round(1)
                alle_hist_temp[key] = daily
                print(f'  Temp historisch: {len(daily):,} Tage')
        except Exception as e:
            print(f'  FEHLER Temp historical: {e}')

        # Niederschlag historisch
        try:
            url = get_zip_url(BASE_RAIN, 'historical', sid)
            if url:
                df = lade_zip(url)
                df['tag'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d').dt.date.astype(str)
                if 'RSK' in df.columns:
                    alle_hist_regen[key] = df[['tag', 'RSK']].rename(columns={'RSK': 'regen_mm'})
                    print(f'  Regen historisch: {len(df):,} Tage')
        except Exception as e:
            print(f'  FEHLER Regen historical: {e}')

        # Klimanormale berechnen
        kn = {}
        if key in alle_hist_temp:
            kn['temp_6190'] = klimanormale_berechnen(alle_hist_temp[key], 'tmax', 1961, 1990)
            kn['temp_9120'] = klimanormale_berechnen(alle_hist_temp[key], 'tmax', 1991, 2020)
        if key in alle_hist_regen:
            df_r = alle_hist_regen[key].copy().sort_values('tag')
            df_r['regen_30d'] = df_r['regen_mm'].rolling(30, min_periods=20).sum().round(1)
            df_r = df_r[['tag', 'regen_30d']].rename(columns={'regen_30d': 'val'})
            kn['regen_6190'] = klimanormale_berechnen(df_r, 'val', 1961, 1990)
            kn['regen_9120'] = klimanormale_berechnen(df_r, 'val', 1991, 2020)
        klimanormale[key] = kn
        print(f'  Klimanormale berechnet')


# ─── SCHRITT 3: Aktuelle Daten (recent) stündlich abrufen ────────
print('\n=== Lade aktuelle Daten (recent) ===')
heute = datetime.today().date()
von_42 = heute - timedelta(days=42)
aktuelle_daten = {}

for key, st in STATIONEN.items():
    sid = st['id']
    print(f"\n{st['name']} (ID {sid})")
    sd = {}

    # Aktuelle Temperaturdaten (stündlich)
    try:
        url = get_zip_url(BASE_TEMP, 'recent', sid)
        if url:
            df = lade_zip(url)
            df['datum'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H')
            df['tag'] = df['datum'].dt.date

            # Tageshöchstwerte aus Stundenwerten
            daily = df.groupby('tag')['TT_TU'].agg(tmax='max', tmean='mean').reset_index()
            daily['tmax'] = daily['tmax'].round(1)
            daily['tmean'] = daily['tmean'].round(1)
            daily['tag'] = daily['tag'].astype(str)

            # Letzter Stundenwert (für "aktuell gerade")
            letzter = df.dropna(subset=['TT_TU']).sort_values('datum').iloc[-1]
            sd['temp_aktuell_stunde'] = round(float(letzter['TT_TU']), 1)
            sd['temp_aktuell_zeit']   = letzter['datum'].strftime('%H:%M Uhr')

            # Nur letzte 42 Tage für Diagramm
            recent = daily[daily['tag'] >= str(von_42)]
            sd['temp_tage'] = recent[['tag', 'tmax', 'tmean']].to_dict(orient='records')

            # Tageshöchstwert heute
            heute_row = daily[daily['tag'] == str(heute)]
            sd['temp_heute_max'] = round(float(heute_row['tmax'].iloc[0]), 1) if not heute_row.empty else None

            # Sommertage (≥25°C) je Jahr seit 1995
            daily['tag'] = pd.to_datetime(daily['tag'])
            daily['jahr'] = daily['tag'].dt.year
            st_df = daily[daily['jahr'] >= 1995].groupby('jahr').apply(
                lambda x: int((x['tmax'] >= 25).sum())
            ).reset_index()
            st_df.columns = ['jahr', 'anzahl']
            sd['sommertage_recent'] = st_df.to_dict(orient='records')

            print(f'  Temp: {len(daily):,} Tage, aktuell {sd["temp_aktuell_stunde"]}°C ({sd["temp_aktuell_zeit"]})')
    except Exception as e:
        print(f'  FEHLER Temp recent: {e}')

    # Aktuelle Niederschlagsdaten (täglich)
    try:
        url = get_zip_url(BASE_RAIN, 'recent', sid)
        if url:
            df = lade_zip(url)
            df['tag'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d').dt.date
            if 'RSK' in df.columns:
                df = df[['tag', 'RSK']].rename(columns={'RSK': 'regen_mm'}).sort_values('tag')
                df['regen_30d'] = df['regen_mm'].rolling(30, min_periods=20).sum().round(1)
                df['tag'] = df['tag'].astype(str)
                recent = df[df['tag'] >= str(von_42)]
                sd['regen_tage'] = recent[['tag', 'regen_30d']].to_dict(orient='records')
                letzter = df.dropna(subset=['regen_30d'])
                sd['regen_30d_aktuell'] = round(float(letzter['regen_30d'].iloc[-1]), 1) if not letzter.empty else None
                print(f'  Regen 30d: {sd.get("regen_30d_aktuell")} mm')
    except Exception as e:
        print(f'  FEHLER Regen recent: {e}')

    aktuelle_daten[key] = sd


# ─── SCHRITT 4: JSON speichern ────────────────────────────────────
output = {
    'meta': {
        'erstellt_am': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'quelle': 'Deutscher Wetterdienst (DWD) Open Data',
        'url': 'https://opendata.dwd.de/',
        'stationen': {k: v['name'] for k, v in STATIONEN.items()}
    },
    'klimanormale': klimanormale,
    'aktuell': aktuelle_daten
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=np_serial)

groesse = os.path.getsize(OUTPUT_FILE) / 1024
print(f'\n✅ {OUTPUT_FILE} gespeichert ({groesse:.0f} KB)')
print(f'   Erstellt: {output["meta"]["erstellt_am"]} UTC')
