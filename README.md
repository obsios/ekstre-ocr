# Runpod PaddleOCR Worker

Bu klasor, banka ekstrelerini PaddleOCR ile okuyup **ham OCR sonucunu** donduren
bagimsiz bir Runpod Serverless worker'idir. Gemini, banka sablonu ve islem
normalizasyonu bu imajda yoktur.

## Neden kalici disk yok?

PP-OCRv6 small modelleri Docker imajina build sirasinda gomulur. Bu, modelin
worker acilirken ag diskinden okunmasindan daha hizlidir. Ayrica network volume
bos dururken de depolama ucreti uretir ve endpoint'i tek veri merkezine baglar.

Maliyet ayari Runpod tarafinda `workersMin=0` olacaktir: istek yokken GPU
kapanir. Ilk istek bir worker acar; imaj ve model onceden hazir oldugu icin
uygulama baslangicinda indirme veya `pip install` yapilmaz.

## Girdi

URL onerilir; Runpod istek boyutu sinirina buyuk PDF'leri Base64 ile tasimayin.

```json
{
  "input": {
    "url": "https://.../ekstre.pdf",
    "filename": "ekstre.pdf",
    "dpi": 300
  }
}
```

Kucuk dosyalar icin `url` yerine `base64` kullanilabilir. PDF, PNG, JPEG ve
Pillow'un okuyabildigi cok kareli goruntuler kabul edilir.

## Cikti

Her sayfa icin su alanlar doner:

- `text`: satir sonlariyla birlestirilmis ham OCR metni
- `words[]`: metin, guven, polygon ve bbox
- `width`, `height`, `ocr_seconds`
- belge SHA-256, sayfa sayisi ve toplam sureler

Birden fazla sayfa ayni OCR modeliyle toplu islenir. `ocr_seconds` toplu islemde
sayfa basina dusen ortalama suredir; `ocr_timing_scope` alani bunu
`batch_average` olarak belirtir. Kesin toplam OCR suresi `timing.ocr_seconds`
alanindadir. Varsayilan sayfa grubu `OCR_PAGE_BATCH_SIZE=8` ile degistirilebilir.

## GitHub Actions ile imaj olusturma

1. Bu klasorun **icerigini** yeni bir GitHub reposunun kokune koyun.
2. `main` dalina push edin.
3. Actions, `linux/amd64` imajini `ghcr.io/KULLANICI/REPO:sha-XXXXXXX`
   etiketiyle olusturur.
4. GitHub Packages ekranindan imaji Public yapin veya Runpod'a GHCR registry
   kimlik bilgisi ekleyin.
5. Runpod template ve endpoint'e daima `sha-...` gibi sabit etiketi verin;
   `latest` kullanmayin.

Yerel Docker gerekmez.

## Runpod endpoint ayarlari

Bir sonraki kurulum adiminda kullanilacak baslangic degerleri:

- Queue-based Serverless endpoint
- GPU: NVIDIA L4 (mevcut testte 14 sayfa toplam yaklasik 22 saniye)
- Active workers / workersMin: `0`
- Max workers: `1` (ilk surum; paralel istek gelirse sonra artirilir)
- Idle timeout: `5` saniye
- Execution timeout: en az `600` saniye
- Container disk: `15 GB`
- Network volume: **yok**
- Data center sabitleme: **yok**

Ilk soguk istek icin `/run` kullanip job durumunu poll edin. Sicak worker icin
`/runsync` kullanilabilir.

## Yerel hafif test

Docker veya GPU gerektirmeyen giris testi:

```bash
python -m unittest discover -s tests -v
```

Gercek OCR testi GitHub Actions imaji tamamlandiktan sonra Runpod'a gercek bir
ekstre gonderilerek yapilmalidir. Endpoint `ready` gorunmesi tek basina yeterli
bir test degildir.
