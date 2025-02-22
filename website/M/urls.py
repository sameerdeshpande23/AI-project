
from django.urls import path,include
from . import views
urlpatterns = [
    path('cam/',views.home,name="h")
]