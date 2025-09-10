from django.contrib import admin
from .models import RecommendDocument,ListLesson, UserAction,UserAction

admin.site.register(RecommendDocument)


class ListLessonAdmin(admin.ModelAdmin):
    list_display = ('lesson_name', 'course', 'description')  # dùng field thực sự có trong ListLesson

admin.site.register(ListLesson, ListLessonAdmin)

@admin.register(UserAction)
class UserActionAdmin(admin.ModelAdmin):
    list_display = ('user', 'video', 'action', 'timestamp')
    list_filter = ('action', 'timestamp')

