from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect, HttpResponse, Http404, HttpResponseBadRequest
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm, AuthenticationForm
from django.core import serializers
from django.contrib.auth.decorators import login_required

def index(request):
    context_dict = {}
    return render ( request, "enular/test.html", context_dict)