"""covipred URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
urlpatterns = [
    path('admin/', admin.site.urls),
    path('covipred/',views.covipred,name="covipred"),
    path('covidtest',views.covidtest,name="covidtest"),
    path('case',views.case,name="case"),
    path('predict',views.predict,name="predict"),
    url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),

    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),

]