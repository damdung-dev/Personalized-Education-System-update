from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'home'

urlpatterns = [
    path('', views.index,name='home'),
    path('account/', views.account_view,name='account'),
    path('courses/', views.courses_view,name='courses'),
    path('calendar/', views.calendar_view,name='calendar'),
    path('chat/', views.chat,name='chat'),
    path('notification/', views.notification_view,name='notifications'),
    path('results/', views.results_view,name='results'),
    path('documents/', views.documents_view,name='documents'),
    path('dashboard/', views.dashboard_view,name='dashboard'),
    path('teachers/', views.teachers, name='teachers'),
    path('courses-current/', views.courses_current, name='courses_current'),
    path('help/', views.help_page, name='help'),
    path('login/', views.login_view, name='exit'),
    path('course/<str:code>/', views.course_detail, name='course_detail'),
    path('register-course/<str:code>/', views.register_course, name='register_course'),
    path('chat/api/', views.chat_api, name='chat_api'),
    path('lesson/<int:lesson_id>/', views.lesson_detail, name='lessons_page'),
    path("lesson/<int:lesson_id>/log-action/", views.log_action, name="log_action"),
    path('youtube_search/', views.youtube_search_view, name='youtube_search'),
    path('log-book-action/', views.log_book_action, name='log_book_action'),
]
#