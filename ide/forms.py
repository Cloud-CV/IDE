from allauth.account.forms import SignupForm
from django import forms

class FabrikSignupForm(SignupForm):
    username.widget.attrs.update({'class': 'validate'})
    email.widget.attrs.update({'class': 'validate'})
    password1.widget.attrs.update({'class': 'validate'})

    def signup(self, request, user):
        user.save()
        return user