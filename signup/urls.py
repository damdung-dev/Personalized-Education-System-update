from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='signup_view'),
    path('home/',views.home_view,name='signup_pressed'),
]