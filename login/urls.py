#from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='login'),
    path('home/',views.home_view,name='home_view'),
    path('signup/',views.signup_view,name='signup_view'),
    path('forgotpassword/',views.forgotpassword_view,name='forgotpassword'),
]