from allauth.account.forms import SignupForm


class FabrikSignupForm(SignupForm):

    def __init__(self, *args, **kwargs):
        super(FabrikSignupForm, self).__init__(*args, **kwargs)

        username = self.fields['username']
        email = self.fields['email']
        password1 = self.fields['password1']
        password2 = self.fields['password2']

        username.widget.attrs.update({'class': 'validate'})
        email.widget.attrs.update({'class': 'validate'})
        password1.widget.attrs.update({'class': 'validate'})

        username.label = "Username"
        email.label = "Email"
        password1.label = "Password"
        password2.label = "Confirm Password"

    def signup(self, request, user):
        user.save()
        return user
