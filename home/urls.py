from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name ='home'),
    path('login/',views.login, name ='login'),
    path('logout/',views.logout, name ='logout'),
    path('model_k/',views.model_k, name ='GradientBoostingRegressor'),
    path('model_url_k/',views.model_url_k, name ='model_url_k')
]
