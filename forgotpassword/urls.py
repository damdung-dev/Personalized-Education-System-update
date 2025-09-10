from django.urls import path
from . import views

app_name = 'forgotpassword' 

urlpatterns = [
    path('', views.index, name='forgotpassword'),
    path('send-email/', views.send_test_email,name='send_test_mail'),
    path('verification-code/',views.confirm_code,name='confirm_code'),
    path('verify-code-button/',views.confirm_code,name='verify'),
]
#