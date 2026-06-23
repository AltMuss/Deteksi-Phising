import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import socket
import ssl
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from urllib.parse import urlparse

# Install requests & beautifulsoup4 jika belum ada
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Phishing Website",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.85; }

    .phishing-result {
        background: linear-gradient(135deg, #fff0f0, #ffe0e0);
        border: 2px solid #e74c3c; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .phishing-result h2 { color: #e74c3c; font-size: 2.2rem; margin: 0; }

    .legit-result {
        background: linear-gradient(135deg, #f0fff4, #d4edda);
        border: 2px solid #27ae60; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .legit-result h2 { color: #27ae60; font-size: 2.2rem; margin: 0; }

    .warning-result {
        background: linear-gradient(135deg, #fffbf0, #fef3cd);
        border: 2px solid #f39c12; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .url-box {
        background: #e8f4fd; border: 2px solid #3498db;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
        font-family: monospace; word-break: break-all;
    }
    .badge-html {
        background: #27ae60; color: white; border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; font-weight: bold;
    }
    .badge-url {
        background: #e67e22; color: white; border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('rf_model.pkl') or not os.path.exists('feature_names.pkl'):
        return None, None, None
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    feat_imp = pd.read_csv('feature_importance.csv') if os.path.exists('feature_importance.csv') else None
    return model, feature_names, feat_imp

model, feature_names, feat_imp = load_model()

# ─── Scraping HTML ────────────────────────────────────────────────────────────
def scrape_html(url, timeout=8):
    """Akses URL dan parse HTML-nya. Return (soup, html_text, error_msg)."""
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36')
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout,
                            allow_redirects=True, verify=False)
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup, resp.text, None
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout,
                                allow_redirects=True, verify=False)
            soup = BeautifulSoup(resp.text, 'html.parser')
            return soup, resp.text, None
        except Exception as e:
            return None, None, str(e)
    except requests.exceptions.ConnectionError:
        return None, None, "Tidak dapat terhubung ke website"
    except requests.exceptions.Timeout:
        return None, None, "Website timeout (terlalu lama merespons)"
    except Exception as e:
        return None, None, str(e)


def extract_html_features(soup, html_text, url):
    """Ekstrak fitur HTML dari BeautifulSoup object."""
    features = {}
    if soup is None:
        # Kembalikan semua 0 kalau gagal scraping
        html_keys = [
            'nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks',
            'ratio_nullHyperlinks','nb_extCSS','ratio_intRedirection',
            'ratio_extRedirection','ratio_intErrors','ratio_extErrors',
            'login_form','external_favicon','links_in_tags','submit_email',
            'ratio_intMedia','ratio_extMedia','sfh','iframe','popup_window',
            'safe_anchor','onmouseover','right_clic','empty_title',
            'domain_in_title','google_index','page_rank','web_traffic',
            'InsecureForms','RelativeFormAction','ExtFormAction',
            'AbnormalFormAction','FakeLinkInStatusBar','RightClickDisabled',
            'PopUpWindow','SubmitInfoToEmail','IframeOrFrame','MissingTitle',
            'ImagesOnlyInForm','ExtFavicon','PctExtHyperlinks',
            'PctExtResourceUrls','PctNullSelfRedirectHyperlinks',
            'ExtMetaScriptLinkRT','PctExtNullSelfRedirectHyperlinksRT',
        ]
        return {k: 0 for k in html_keys}, False

    parsed = urlparse(url)
    domain = parsed.netloc

    # ── Hyperlinks ─────────────────────────────────────────────────────────────
    all_links   = soup.find_all('a', href=True)
    total_links = len(all_links)

    int_links  = sum(1 for a in all_links if domain in a['href'])
    ext_links  = sum(1 for a in all_links if domain not in a['href'] and a['href'].startswith('http'))
    null_links = sum(1 for a in all_links if a['href'] in ['#', '', 'javascript:void(0)', 'javascript:;'])

    features['nb_hyperlinks']        = total_links
    features['ratio_intHyperlinks']  = int_links  / max(total_links, 1)
    features['ratio_extHyperlinks']  = ext_links  / max(total_links, 1)
    features['ratio_nullHyperlinks'] = null_links / max(total_links, 1)
    features['PctExtHyperlinks']     = features['ratio_extHyperlinks']
    features['PctNullSelfRedirectHyperlinks'] = features['ratio_nullHyperlinks']
    features['PctExtNullSelfRedirectHyperlinksRT'] = (ext_links + null_links) / max(total_links, 1)

    # ── CSS External ───────────────────────────────────────────────────────────
    css_links = soup.find_all('link', rel=lambda r: r and 'stylesheet' in r)
    ext_css   = sum(1 for c in css_links if c.get('href','').startswith('http') and domain not in c.get('href',''))
    features['nb_extCSS'] = ext_css

    # ── Media ──────────────────────────────────────────────────────────────────
    imgs       = soup.find_all('img', src=True)
    int_media  = sum(1 for i in imgs if domain in i['src'] or not i['src'].startswith('http'))
    ext_media  = sum(1 for i in imgs if i['src'].startswith('http') and domain not in i['src'])
    total_media = max(len(imgs), 1)
    features['ratio_intMedia'] = int_media / total_media
    features['ratio_extMedia'] = ext_media / total_media

    # ── Forms ──────────────────────────────────────────────────────────────────
    forms = soup.find_all('form')
    login_keywords = ['login', 'signin', 'password', 'passwd', 'user', 'email']
    has_login_form = any(
        any(kw in str(f).lower() for kw in login_keywords)
        for f in forms
    )
    features['login_form']      = 1 if has_login_form else 0
    features['InsecureForms']   = 1 if any(f.get('action','').startswith('http://') for f in forms) else 0
    features['RelativeFormAction'] = 1 if any(not f.get('action','').startswith('http') and f.get('action','') not in ['','#'] for f in forms) else 0
    features['ExtFormAction']   = 1 if any(f.get('action','').startswith('http') and domain not in f.get('action','') for f in forms) else 0
    features['AbnormalFormAction'] = 1 if any(f.get('action','') in ['#','about:blank','javascript:true'] for f in forms) else 0
    features['sfh']             = 1 if any(f.get('action','') in ['','#'] for f in forms) else 0

    # Cek submit ke email
    has_submit_email = bool(re.search(r'mailto:', str(soup)))
    features['submit_email']    = 1 if has_submit_email else 0
    features['SubmitInfoToEmail'] = features['submit_email']

    # ── Favicon ────────────────────────────────────────────────────────────────
    favicon = soup.find('link', rel=lambda r: r and 'icon' in ' '.join(r).lower() if r else False)
    ext_favicon = 0
    if favicon and favicon.get('href','').startswith('http') and domain not in favicon.get('href',''):
        ext_favicon = 1
    features['external_favicon'] = ext_favicon
    features['ExtFavicon']       = ext_favicon

    # ── Script / Links in tags ─────────────────────────────────────────────────
    scripts = soup.find_all('script', src=True)
    ext_scripts = sum(1 for s in scripts if s['src'].startswith('http') and domain not in s['src'])
    total_tags  = max(len(scripts) + len(css_links), 1)
    features['links_in_tags']       = ext_scripts / total_tags
    features['ExtMetaScriptLinkRT'] = ext_scripts / total_tags

    # ── Iframe ─────────────────────────────────────────────────────────────────
    iframes = soup.find_all('iframe')
    features['iframe']       = 1 if iframes else 0
    features['IframeOrFrame'] = features['iframe']

    # ── Popup / onmouseover / right click ──────────────────────────────────────
    html_str = str(soup)
    features['popup_window']      = 1 if 'window.open' in html_str else 0
    features['PopUpWindow']       = features['popup_window']
    features['onmouseover']       = 1 if 'onmouseover' in html_str.lower() else 0
    features['right_clic']        = 1 if 'contextmenu' in html_str.lower() or 'event.button==2' in html_str else 0
    features['RightClickDisabled'] = features['right_clic']
    features['FakeLinkInStatusBar'] = 1 if 'window.status' in html_str else 0

    # ── Safe anchor ────────────────────────────────────────────────────────────
    unsafe = sum(1 for a in all_links if a.get('href','') in ['#', 'javascript:;', 'javascript:void(0)'])
    features['safe_anchor'] = unsafe / max(total_links, 1)

    # ── Title ──────────────────────────────────────────────────────────────────
    title_tag = soup.find('title')
    title_txt = title_tag.get_text().strip() if title_tag else ''
    features['empty_title']  = 1 if not title_txt else 0
    features['MissingTitle'] = features['empty_title']
    features['domain_in_title'] = 1 if domain.replace('www.','') in title_txt.lower() else 0

    # ── Images only in form ────────────────────────────────────────────────────
    features['ImagesOnlyInForm'] = 0
    if forms:
        form_html = str(forms[0])
        has_text_input = bool(re.search(r'<input[^>]*type=["\']text', form_html, re.I))
        has_img        = bool(re.search(r'<img', form_html, re.I))
        features['ImagesOnlyInForm'] = 1 if (has_img and not has_text_input) else 0

    # Redirect ratio (estimasi dari meta refresh)
    meta_refresh = soup.find_all('meta', attrs={'http-equiv': re.compile('refresh', re.I)})
    features['ratio_intRedirection'] = 0.0
    features['ratio_extRedirection'] = 1.0 if meta_refresh else 0.0
    features['ratio_intErrors']      = 0.0
    features['ratio_extErrors']      = 0.0

    # External resource URLs ratio
    all_srcs    = [t.get('src','') for t in soup.find_all(src=True)]
    ext_srcs    = sum(1 for s in all_srcs if s.startswith('http') and domain not in s)
    features['PctExtResourceUrls']   = ext_srcs / max(len(all_srcs), 1)
    features['PctExtResourceUrlsRT'] = features['PctExtResourceUrls']

    # Default fitur yang tidak bisa diestimasi
    features['google_index'] = 0
    features['page_rank']    = 0
    features['web_traffic']  = 0

    return features, True


# ─── Ekstrak Fitur URL ────────────────────────────────────────────────────────
def extract_url_features(url):
    features = {}
    if not url.startswith('http'):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path   = parsed.path
        query  = parsed.query
    except:
        domain, path, query = url, '', ''

    features['length_url']         = len(url)
    features['length_hostname']    = len(domain)
    features['length_path']        = len(path)
    features['length_url_path']    = len(url.split('?')[0])
    features['nb_dots']            = url.count('.')
    features['nb_hyphens']         = url.count('-')
    features['nb_at']              = url.count('@')
    features['nb_qm']              = url.count('?')
    features['nb_and']             = url.count('&')
    features['nb_or']              = url.count('|')
    features['nb_eq']              = url.count('=')
    features['nb_underscore']      = url.count('_')
    features['nb_tilde']           = url.count('~')
    features['nb_percent']         = url.count('%')
    features['nb_slash']           = url.count('/')
    features['nb_star']            = url.count('*')
    features['nb_colon']           = url.count(':')
    features['nb_comma']           = url.count(',')
    features['nb_semicolumn']      = url.count(';')
    features['nb_dollar']          = url.count('$')
    features['nb_space']           = url.count(' ')
    features['nb_www']             = url.lower().count('www')
    features['nb_com']             = url.lower().count('.com')
    features['nb_dslash']          = url.count('//')
    features['NumDash']            = url.count('-')
    features['NumDashInHostname']  = domain.count('-')
    features['NumNumericChars']    = sum(c.isdigit() for c in url)
    features['NumUnderscore']      = url.count('_')
    features['NumPercent']         = url.count('%')
    features['NumQueryComponents'] = len(query.split('&')) if query else 0
    features['NumAmpersand']       = url.count('&')
    features['ratio_digits_url']   = sum(c.isdigit() for c in url) / max(len(url), 1)
    features['ratio_digits_host']  = sum(c.isdigit() for c in domain) / max(len(domain), 1)

    ip_pat = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    features['ip']        = 1 if ip_pat.search(domain) else 0
    features['IpAddress'] = features['ip']
    features['port']      = 1 if ':' in domain.split('.')[-1] else 0
    features['https_token']     = 1 if parsed.scheme == 'https' else 0
    features['HttpsInHostname'] = 1 if 'https' in domain.lower() else 0
    features['punycode']        = 1 if 'xn--' in url.lower() else 0

    path_parts = [p for p in path.split('/') if p]
    features['path_extension']   = 1 if (path and '.' in path.split('/')[-1]) else 0
    features['PathLevel']        = len(path_parts)
    features['longest_word_path']= max((len(w) for w in re.split(r'[/\-_.]', path) if w), default=0)
    features['avg_word_path']    = np.mean([len(w) for w in re.split(r'[/\-_.]', path) if w]) if path else 0

    domain_parts = domain.replace('www.', '').split('.')
    features['nb_subdomains']  = max(len(domain_parts) - 2, 0)
    features['NumSubDomains']  = features['nb_subdomains']

    tld = domain_parts[-1] if domain_parts else ''
    features['tld_in_path']       = 1 if tld in path.lower() else 0
    features['tld_in_subdomain']  = 1 if len(domain_parts) > 2 and tld in '.'.join(domain_parts[:-2]) else 0
    features['abnormal_subdomain']= 1 if bool(re.search(r'\d+\.\d+', domain)) else 0

    suspicious_words = ['secure','account','update','login','signin','bank',
                        'verify','password','confirm','paypal','ebay','amazon']
    features['phish_hints']        = sum(1 for w in suspicious_words if w in url.lower())
    features['brand_in_subdomain'] = 1 if any(w in '.'.join(domain_parts[:-2]).lower() for w in suspicious_words) else 0
    features['brand_in_path']      = 1 if any(w in path.lower() for w in suspicious_words) else 0
    features['suspecious_tld']     = 1 if tld in ['tk','ml','ga','cf','gq','xyz','pw','top','click'] else 0

    shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd','buff.ly']
    features['shortening_service'] = 1 if any(s in domain.lower() for s in shorteners) else 0

    try:
        socket.setdefaulttimeout(3)
        socket.gethostbyname(domain.split(':')[0])
        features['dns_record']    = 1
        features['DNSRecordType'] = 1
    except:
        features['dns_record']    = 0
        features['DNSRecordType'] = 0

    try:
        ctx  = ssl.create_default_context()
        conn = ctx.wrap_socket(socket.socket(), server_hostname=domain.split(':')[0])
        conn.settimeout(3)
        conn.connect((domain.split(':')[0], 443))
        conn.close()
        features['domain_with_copyright'] = 1
    except:
        features['domain_with_copyright'] = 0

    features['NumSensitiveWords']  = features['phish_hints']
    features['EmbeddedBrandName']  = features['brand_in_path']
    features['SubdomainLevelRT']   = features['nb_subdomains']
    features['UrlLengthRT']        = len(url)
    features['HostnameLength']     = len(domain)
    features['QueryLength']        = len(query)
    features['DoubleSlashInPath']  = 1 if '//' in path else 0
    features['domain_age']         = 0

    return features


def align_features(raw_features, feature_names):
    row = {feat: raw_features.get(feat, 0) for feat in feature_names}
    return pd.DataFrame([row])


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔐 Deteksi Phishing Website</h1>
    <p>Random Forest + 129 Fitur URL & HTML + Explainable AI | Akurasi 97.48%</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ File `rf_model.pkl` tidak ditemukan! Pastikan file model ada di folder yang sama.")
    st.stop()

# Install warning
if not SCRAPING_AVAILABLE:
    st.warning("⚠️ Library `requests` dan `beautifulsoup4` belum terinstall. Jalankan: `pip install requests beautifulsoup4`")

# ─── Input URL ────────────────────────────────────────────────────────────────
st.markdown("### 🔗 Masukkan URL Website")

col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input("",
        placeholder="Contoh: https://www.google.com atau http://suspicious-login.xyz/verify",
        label_visibility="collapsed")
with col2:
    predict_btn = st.button("🔍 Deteksi", type="primary", use_container_width=True)

st.caption("Contoh URL:")
ec1, ec2, ec3, ec4 = st.columns(4)
if ec1.button("✅ google.com"):       url_input = "https://www.google.com"
if ec2.button("✅ github.com"):       url_input = "https://github.com"
if ec3.button("🚨 paypal-verify.tk"): url_input = "http://paypal-verify.tk/account/login"
if ec4.button("🚨 secure-bank.xyz"):  url_input = "http://secure-bank-login.xyz/update/password"

# ─── Prediksi ─────────────────────────────────────────────────────────────────
if (predict_btn or url_input) and url_input.strip():
    url = url_input.strip()
    if not url.startswith('http'):
        url = 'https://' + url

    # Progress
    progress = st.progress(0, "🔍 Mengekstrak fitur URL...")
    url_feats = extract_url_features(url)
    progress.progress(30, "🌐 Mengakses konten HTML website...")

    html_success = False
    html_error   = None
    if SCRAPING_AVAILABLE:
        import urllib3
        urllib3.disable_warnings()
        soup, html_text, err = scrape_html(url)
        if soup:
            html_feats, html_success = extract_html_features(soup, html_text, url)
            progress.progress(80, "🤖 Menjalankan model prediksi...")
        else:
            html_feats, _ = extract_html_features(None, None, url)
            html_error    = err
            progress.progress(70, "⚠️ HTML tidak dapat diakses, lanjut dengan fitur URL...")
    else:
        html_feats = {k: 0 for k in [
            'nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks',
            'ratio_nullHyperlinks','nb_extCSS','ratio_intRedirection',
            'ratio_extRedirection','ratio_intErrors','ratio_extErrors',
            'login_form','external_favicon','links_in_tags','submit_email',
            'ratio_intMedia','ratio_extMedia','sfh','iframe','popup_window',
            'safe_anchor','onmouseover','right_clic','empty_title',
            'domain_in_title','google_index','page_rank','web_traffic',
        ]}

    # Gabungkan semua fitur
    all_feats = {**url_feats, **html_feats}
    df_input  = align_features(all_feats, feature_names)

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    progress.progress(100, "✅ Selesai!")
    progress.empty()

    # ── Tampilkan Hasil ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Hasil Analisis")

    # Badge mode
    if html_success:
        st.markdown('<span class="badge-html">✅ Analisis Lengkap: URL + HTML</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-url">⚠️ Analisis Parsial: URL saja</span>', unsafe_allow_html=True)
        if html_error:
            st.caption(f"HTML tidak dapat diakses: {html_error}")

    st.markdown(f'<div class="url-box">🔗 <b>{url}</b></div>', unsafe_allow_html=True)

    # Verdict
    if pred == 1:
        st.markdown(f"""<div class="phishing-result">
            <h2>🚨 PHISHING TERDETEKSI!</h2>
            <p style="font-size:1.2rem;margin:0.8rem 0 0 0;color:#c0392b">
                Probabilitas Phishing: <b>{prob*100:.2f}%</b>
            </p>
            <p style="margin:0.5rem 0 0 0;color:#7f1d1d">
                ⚠️ Jangan masukkan data pribadi di website ini!
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="legit-result">
            <h2>✅ LEGITIMATE</h2>
            <p style="font-size:1.2rem;margin:0.8rem 0 0 0;color:#1e8449">
                Probabilitas Phishing: <b>{prob*100:.2f}%</b>
            </p>
            <p style="margin:0.5rem 0 0 0;color:#145a32">
                ✅ Website ini tampaknya aman.
            </p>
        </div>""", unsafe_allow_html=True)

    # Confidence bar
    fig, ax = plt.subplots(figsize=(8, 1.2))
    color = '#e74c3c' if prob > 0.5 else '#27ae60'
    ax.barh([''], [prob],       color=color,     height=0.5)
    ax.barh([''], [1 - prob],   left=[prob],     color='#ecf0f1', height=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5)
    ax.text(prob / 2, 0, f'{prob*100:.1f}% Phishing',
            ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Confidence Score', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Detail fitur
    with st.expander("🔬 Detail Fitur yang Diekstrak"):
        parsed = urlparse(url)
        domain = parsed.netloc

        st.markdown("**📌 Fitur URL**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Panjang URL",     url_feats.get('length_url', 0))
        c2.metric("Jumlah Titik",    url_feats.get('nb_dots', 0))
        c3.metric("Subdomain",       url_feats.get('nb_subdomains', 0))
        c4.metric("HTTPS",           "✅ Ya" if url_feats.get('https_token', 0) else "❌ Tidak")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("IP Address",          "Ya" if url_feats.get('ip', 0) else "Tidak")
        c6.metric("Kata Mencurigakan",   url_feats.get('phish_hints', 0))
        c7.metric("TLD Mencurigakan",    "Ya" if url_feats.get('suspecious_tld', 0) else "Tidak")
        c8.metric("DNS Record",          "✅ Ada" if url_feats.get('dns_record', 0) else "❌ Tidak")

        if html_success:
            st.markdown("**📌 Fitur HTML**")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total Hyperlinks",   html_feats.get('nb_hyperlinks', 0))
            h2.metric("Login Form",         "Ya" if html_feats.get('login_form', 0) else "Tidak")
            h3.metric("Iframe",             "Ya" if html_feats.get('iframe', 0) else "Tidak")
            h4.metric("Ext. Favicon",       "Ya" if html_feats.get('external_favicon', 0) else "Tidak")

            h5, h6, h7, h8 = st.columns(4)
            h5.metric("Popup Window",       "Ya" if html_feats.get('popup_window', 0) else "Tidak")
            h6.metric("Right Click Off",    "Ya" if html_feats.get('right_clic', 0) else "Tidak")
            h7.metric("Submit ke Email",    "Ya" if html_feats.get('submit_email', 0) else "Tidak")
            h8.metric("Title Kosong",       "Ya" if html_feats.get('empty_title', 0) else "Tidak")

elif predict_btn and not url_input.strip():
    st.warning("⚠️ Masukkan URL terlebih dahulu!")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85rem">
    🔐 Deteksi Phishing Website | Random Forest 97.48% | Universitas Negeri Semarang
</div>
""", unsafe_allow_html=True)
