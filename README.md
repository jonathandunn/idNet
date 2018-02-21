lidNet
--------

To load a model:

	import lidNet
	lid = lidNet("Path to model file")
	
Note: The model is too big for GitHub:  	https://s3.amazonaws.com/jonathandunn/Model.LID.464langs.50chars.130k_hash.3-3.Callback.hdf
	
Once a model is loaded, it has the following properties:

	lid.n_features		# Number of features in the model
	lid.n_classes		# Number of languages in the model
	lid.lang_mappings	# Dictionary of {"iso_code": "language_name"} mappings for all ISO 639-3 codes
	lid.langs		# List of ISO 639-3 codes for languages present in the current model
	
Loaded models perform the following tasks:

	lid.predict(data)		# Takes an array of strings and returns an array of predicted language codes. Also takes individual strings.
	
The *predict* method requires at least two processes: one to extract features from the input and one to evaluate the model using TensorFlow. Ideally, many cores are available for TensorFlow; whatever cores are available will be used. Performance is best with batches of ~2k instances.

lidNet.py can be used on its own with pre-trained models. Note that the model is too large for GitHub.

Training New Models
----------------------

To train new models, the training data needs to be prepared. This process is automated; see the *Data* directory for directions and scripts.

Once the data is prepared, modify the parameters in lidNet_Train.py run it: *python lidNet_Train.py*;  Or, import and call the function directly: *train(PARAMETERS)*

Parameters
-----------

feature_type 
			
	(str): "hashing" for character n-gram hashes; "tensor" for character sequence tensors

model_type	
			
	(str): "mlp" is a dense multi-layered perceptron; "cnn" uses convolutional layers

filename_prefix		
	
	(str): A prefix for all file names; allows communication with non-local storage

s3_storage	
			
	(boolean/str): If not False, use boto3 to connect to data on s3 bucket, with bucket name as string

prefix	
				
	(str): If using s3 bucket, this is prefix for data folder

nickname
				
	(str): The nickname for saving / loading models

n_features	
			
	(int): Number of character n-gram features to allow, hashing only

n_gram_tuple
			
	(tuple of ints): Range of n-grams to hash, hashing only

line_length	
			
	(int): For tensor features, size of observations (i.e., number of characters per observation)

max_chars 	
			
	(int): The maximum number of characters for tensor represetations; limits dimensionality

divide_data		
		
	(boolean): If True, crawl for dataset; if False, just load it

lang_threshold	
		
	(int): Number of samples in the dataset required to include a language; can come from single domain

domain_threshold	
	
	(int): Number of domains in the dataset require to include a language; samples per domain doesn't matter

n_samples		
		
	(int): Number of samples of each language+domain to use per epoch (currently, 18 possible domains)

n_concat 			
	
	(int): Number of samples to concat per batch, to control memory use

data_dir			
	
	(str): Path to data directory; contains domain directories with language sub-directories

load_vectorizer		
	
	(boolean): If using tensors, load a previously fit vectorizer for the character inventory; tensor only

vectorizer_file
			
	(str): If loading a charcter inventory, this is the filename to use

file_list	
			
	(str): If a string, file to load for list of training files (as a text file)

write_files	
			
	(boolean): If True, write the file list to the specified file

n_workers	
			
	(int): Number of workers for Keras fit function

q_size	
				
	(int): Number of samples to queue for Keras fit function

pickle_safe				
	
	(boolean): Whether Keras multi-processing is pickle safe; it is on Linux but not on Windows
	

Environment Requirements
-------------------------

lidNet requires TensorFlow (currently tested on 1.0 and 1.4), Keras (2.0), and scikit-learn (0.18)


Current Languages
-------------------

aai: Arifama-Miniafia

aak: Ankave

aau: Abau

abt: Ambulas

abx: Inabaknon

aby: Aneme Wake

acr: Achi

acu: Achuar-Shiwiar

aey: Amele

afr: Afrikaans

agd: Agarabi

agg: Angor

agm: Angaataha

agr: Aguaruna

agu: Aguacateco

aia: Arosi

ake: Akawaio

alp: Alune

alq: Algonquin

ame: Yanesha'

amh: Amharic

amk: Ambai

amm: Ama

amn: Amanab

amp: Alamblak

amr: Amarakaeri

amu: Amuzgo

anv: Denya

aoj: Mufian

aom: Ömie

aon: Bumbita Arapesh

ape: Bukiyip

apn: Apinayé

apr: Arop-Lokep

apu: Apurinã

apy: Apalaí

apz: Safeyoka

ara: Arabic

arg: Aragonese

arl: Arabela

arn: Mapudungun

asm: Assamese

aso: Dano

ata: Pele-Ata

att: Pamplona Atta

auc: Waorani

avt: Au

awb: Awa

azd: Nahuatl

aze: Azerbaijani

bao: Waimaha

bbb: Barai

bbr: Girawa

bch: Bariai

bcl: Central Bikol

bdd: Bunama

bef: Benabena

bel: Belarusian

ben: Bengali

bgs: Tagabawa

big: Biangai

bjr: Binumarien

bjv: Bedjond

bkq: Bakairí

blw: Balangao

blz: Balantak

bmh: Kein

bmu: Somba-Siawari

bnc: Bontok

bnp: Bola

bod: Tibetan

boj: Anjam

bon: Bine

box: Buamu

bqc: Boko

bqp: Busa

bsn: Barasana-Eduria

bss: Akoose

buk: Bugawac

bul: Bulgarian

bus: Bokobaru

byr: Baruya

byx: Qaqet

bzd: Bribri

caa: Chortí

cak: Kaqchikel

cao: Chácobo

cap: Chipaya

cat: Catalan

cax: Chiquitano

cbc: Carapana

cbi: Chachi

cbr: Cashibo-Cacataibo

cbs: Cashinahua

cbt: Chayahuita

cbu: Candoshi-Shapra

cbv: Cacua

cco: Chinantec

ceb: Cebuano

ces: Czech

cgc: Kagayanen

chf: Tabasco Chontal

chv: Chuvash

cjv: Chuave

cme: Cerma

cni: Asháninka

coe: Koreguaje

cot: Caquinte

cpc: Ajyíninka Apurucayali

crn: El Nayar Cora

crx: Carrier

csb: Kashubian

cub: Cubeo

cui: Cuiba

cut: Teutila Cuicatec

cym: Welsh

dah: Gwahatike

dan: Danish

ded: Dedua

des: Desano

deu: German

diq: Dimli

div: Dhivehi

dje: Zarma

dob: Dobu

dop: Lukpa

dwr: Dawro

dww: Dawawa

dzo: Dzongkha

ell: Greek

emp: Northern Emberá

eng: English

enq: Enga

epo: Esperanto

ese: Ese Ejja

est: Estonian

eus: Basque

faa: Fasu

fai: Faiwol

fao: Faroese

fas: Persian

fin: Finnish

for: Fore

fra: French

gah: Alekano

gaw: Nobonob

gbi: Galela

gdn: Umanakaina

gdr: Wipi

gfk: Patpatar

gle: Irish

glg: Galician

gng: Ngangam

gom: Goan Konkani

gub: Guajajára

guc: Wayuu

gug: Paraguayan Guaraní

guh: Guahibo

guj: Gujarati

gum: Guambiano

guo: Guayabero

gvc: Guanano

gvf: Golin

gwi: Gwichʼin

gym: Ngäbere

hat: Haitian

hbs: Serbo-Croatian

heb: Hebrew

hin: Hindi

hix: Hixkaryána

hla: Halia

hmo: Hiri Motu

hot: Hote

hui: Huli

hun: Hungarian

hus: Huastec

huu: Huitoto

huv: San Mateo Del Mar Huave

hye: Armenian

ian: Iatmul

ido: Ido

ign: Ignaciano

ilo: Iloko

inb: Inga

ind: Indonesian

ino: Inoke-Yate

iou: Tuma-Irumu

ipi: Ipili

isl: Icelandic

ita: Italian

iws: Sepik Iwam

ixl: Ixil

jae: Yabem

jav: Javanese

jic: Tol

jiv: Shuar

jpn: Japanese

kab: Kabyle

kal: Kalaallisut

kan: Kannada

kaq: Capanahua

kat: Georgian

kaz: Kazakh

kbc: Kadiwéu

kbh: Camsá

kek: Kekchí

ken: Kenyang

kew: Kewa

kgk: Kaiwá

kgp: Kaingang

khm: Central Khmer

khz: Keapara

kin: Kinyarwanda

kir: Kirghiz

kje: Kisar

kmo: Kwoma

kms: Kamasau

kmu: Kanite

knv: Tabo

kor: Korean

kos: Kosraean

kpr: Korafe-Yegha

kpw: Kobon

ksd: Kuanua

kto: Kuot

kue: Kuman

kup: Kunimaipa

kur: Kurdish

kvn: Border Kuna

kwj: Kwanga

kyc: Kyaka

kyg: Keyagana

kyq: Kenga

kyz: Kayabí

lac: Lacandon

lao: Lao

lav: Latvian

lcm: Tungag

leu: Kara

lex: Luang

lez: Lezghian

lid: Nyindrou

lit: Lithuanian

lus: Lushai

maa: Mazatec

mal: Malayalam

mam: Mam

mar: Marathi

mav: Sateré-Mawé

maz: Central Mazahua

mbh: Mangseng

mbj: Nadëb

mbl: Maxakalí

mcb: Machiguenga

mcd: Sharanahua

mcf: Matsés

med: Melpa

mee: Mengen

mek: Mekeo

meq: Merey

meu: Motu

mhl: Mauwake

mhr: Mari

mkd: Macedonian

mlg: Malagasy

mlh: Mape

mlp: Bargam

mmo: Buang

mmx: Madak

mna: Mbula

mon: Mongolian

mop: Mopán Maya

mox: Molima

mps: Dadibi

mpt: Mian

mpx: Misima-Panaeati

mqb: Mbuko

mri: Maori

msa: Malay

msy: Aruamu

mux: Bo-Ung

muy: Muyang

mva: Manam

mvn: Minaveha

mxp: Mixe

mya: Burmese

myu: Mundurukú

myy: Macuna

mzn: Mazanderani

nab: Southern Nambikuára

naf: Nabak

nak: Nakanai

nap: Neapolitan

nas: Naasioi

ncu: Chumburung

nep: Nepali

new: Newari

nhu: Noone

nii: Nii

nld: Dutch

noa: Woun Meu

nop: Numanggang

nor: Norwegian

not: Nomatsiguenga

nou: Ewage-Notu

nsn: Nehan

ntp: Tepehuan

nvm: Namiae

ong: Olo

ons: Ono

opm: Oksapmin

ori: Oriya

osi: Osing

ote: Mezquital Otomi

pab: Parecís

pad: Paumarí

pah: Tenharim

pan: Panjabi

pbc: Patamona

pck: Paite Chin

pio: Piapoco

pir: Piratapuyo

pls: San Marcos Tlacoyalco Popoloca

pms: Piemontese

poh: Poqomchi'

poi: Highland Popoluca

pol: Polish

pon: Pohnpeian

por: Portuguese

ppo: Folopa

ptp: Patep

ptu: Bambam

pus: Pashto

pwg: Gapapaiwa

quc: K'iche'

que: Quechua

rai: Ramoaaina

ram: Canela

rkb: Rikbaktsa

ron: Romanian

roo: Rotokas

rro: Waima

rus: Russian

rwo: Rawa

sab: Buglere

sah: Yakut

scn: Sicilian

sey: Secoya

sgs: Samogitian

sgz: Sursurunga

sim: Mende

sin: Sinhala

sja: Epena

slk: Slovak

sll: Salt-Yui

slv: Slovenian

sme: Northern Sami

smk: Bolinao

sna: Shona

snc: Sinaugoro

snd: Sindhi

snn: Siona

snp: Siane

sny: Saniyo-Hiyewe

som: Somali

soq: Kanasi

sot: Southern Sotho

spa: Spanish

spl: Selepet

sps: Saposa

spy: Sabaot

sqi: Albanian

srd: Sardinian

sri: Siriano

srm: Saramaccan

srq: Sirionó

ssd: Siroi

ssx: Samberigi

stq: Saterfriesisch

sua: Sulka

sue: Suena

suz: Sunwar

swa: Swahili

swe: Swedish

swp: Suau

sxb: Suba

szl: Silesian

tam: Tamil

tar: Tarahumara

tat: Tatar

tav: Tatuyo

tbc: Takia

tbg: Tairora

tbl: Tboli

tbz: Ditammari

tca: Ticuna

tel: Telugu

ter: Tereno

tgk: Tajik

tgl: Tagalog

tha: Thai

tif: Tifal

tim: Timbe

tlf: Telefol

toc: Totonac

toj: Tojolabal

tpi: Tok Pisin

tpz: Tinputz

ttc: Tektiteko

tte: Bwanabwana

tuc: Mutu

tuf: Central Tunebo

tuk: Turkmen

tuo: Tucano

tur: Turkish

txu: Kayapó

tzj: Tz'utujil

ubu: Umbu-Ungu

udm: Udmurt

udu: Uduk

uig: Uighur

ukr: Ukrainian

urb: Urubú-Kaapor

urd: Urdu

usa: Usarufa

usp: Uspanteco

uvl: Lote

uzb: Uzbek

vie: Vietnamese

vol: Volapük

vro: Võro

waj: Waffa

wal: Wolaytta

wap: Wapishana

wiu: Wiru

wln: Walloon

wnc: Wantoat

wol: Wolof

wos: Hanga Hundi

wrs: Waris

wsk: Waskia

xav: Xavánte

xho: Xhosa

xla: Kamula

xmf: Mingrelian

xsi: Sio

yaa: Yaminahua

yad: Yagua

yaq: Yaqui

yby: Yaweyuha

yid: Yiddish

yle: Yele

yml: Iamalele

yon: Yongkom

yor: Yoruba

yss: Yessan-Mayo

yuj: Karkar-Yuri

yut: Yopno

yuw: Yau

yva: Yawa

zap: Zapotec

zho: Chinese

zia: Zia

zul: Zulu