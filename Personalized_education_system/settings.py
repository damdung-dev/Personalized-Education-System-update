from pathlib import Path
from django.utils.translation import gettext_lazy as _
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-q_bv$9^5n3l%epp!9j_8_tt)4tvllk3o15q^7r4bwr3^h+%^a+'
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', '45648999ec05.ngrok-free.app']
CSRF_TRUSTED_ORIGINS = ['https://45648999ec05.ngrok-free.app']

# Applications
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'signup',
    'login',
    'forgotpassword',
    'home',
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',   # ✅ Important for i18n
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'Personalized_education_system.urls'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'Personalized_education_system.wsgi.application'

# Database (MySQL)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": "education_system",
        "USER": "root",
        "PASSWORD": "Dung@2292001",
        "HOST": "localhost",
        "PORT": "3306",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# Internationalization
LANGUAGE_CODE = 'en-us'   # default
TIME_ZONE = 'Asia/Ho_Chi_Minh'
USE_I18N = True
USE_L10N = True
USE_TZ = False

# Kho ngôn ngữ (~100 phổ biến)
LANGUAGE_CODE = 'en'   # default

LANGUAGES = [
    ('af', _('Afrikaans')),
    ('am', _('Amharic')),
    ('ar', _('Arabic')),
    ('az', _('Azerbaijani')),
    ('be', _('Belarusian')),
    ('bg', _('Bulgarian')),
    ('bn', _('Bengali')),
    ('bs', _('Bosnian')),
    ('ca', _('Catalan')),
    ('cs', _('Czech')),
    ('cy', _('Welsh')),
    ('da', _('Danish')),
    ('de', _('German')),
    ('el', _('Greek')),
    ('en', _('English')),          # hợp lệ
    ('en-gb', _('English (UK)')),      # thêm en_GB
    ('en-us', _('English (US)')),      # thêm en_US
    ('es', _('Spanish')),
    ('es-mx', _('Spanish (Mexico)')),  # thêm es_MX
    ('et', _('Estonian')),
    ('eu', _('Basque')),
    ('fa', _('Persian')),
    ('fi', _('Finnish')),
    ('fil', _('Filipino')),
    ('fr', _('French')),
    ('fr-ca', _('French (Canada)')),   # thêm fr_CA
    ('he', _('Hebrew')),
    ('hi', _('Hindi')),
    ('hr', _('Croatian')),
    ('hu', _('Hungarian')),
    ('id', _('Indonesian')),
    ('is', _('Icelandic')),
    ('it', _('Italian')),
    ('ja', _('Japanese')),
    ('ko', _('Korean')),
    ('lo', _('Lao')),
    ('lt', _('Lithuanian')),
    ('lv', _('Latvian')),
    ('mk', _('Macedonian')),
    ('ms', _('Malay')),
    ('my', _('Burmese')),
    ('ne', _('Nepali')),
    ('nl', _('Dutch')),
    ('no', _('Norwegian')),
    ('pl', _('Polish')),
    ('pt', _('Portuguese')),       # hợp lệ
    ('ro', _('Romanian')),
    ('ru', _('Russian')),
    ('si', _('Sinhala')),
    ('sk', _('Slovak')),
    ('sl', _('Slovenian')),
    ('sr', _('Serbian')),
    ('sv', _('Swedish')),
    ('sw', _('Swahili')),
    ('ta', _('Tamil')),
    ('th', _('Thai')),
    ('tr', _('Turkish')),
    ('uk', _('Ukrainian')),
    ('ur', _('Urdu')),
    ('uz', _('Uzbek')),
    ('vi', _('Vietnamese')),
    # ... các ngôn ngữ khác ...
    ('zh-cn', _('Chinese (Simplified, CN)')),
    ('zh-tw', _('Chinese (Traditional, TW)')),
]


LOCALE_PATHS = [BASE_DIR / "locale"]

# Static & Media
STATIC_URL = '/static/'

STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / "media"

# Email backend (for production use SMTP, for dev you can use console backend)
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# SMTP settings
EMAIL_HOST = "smtp.gmail.com"      # For Gmail
EMAIL_PORT = 587
EMAIL_USE_TLS = True

# Your email credentials
EMAIL_HOST_USER = "damducdung2001@gmail.com"
EMAIL_HOST_PASSWORD = "eoak kdui rmrh lssc"  # Use App Password if Gmail (not your login password!)

# Default sender (optional but recommended)
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER
