import React from "react";
import DjangoCSRFToken from "django-react-csrftoken";
import $ from "jquery";

class Signup extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      error_username: 0,
      error_email: 0,
      error_password: 0,
      error_confirm_password: 0
    };
    this.submitForm = this.submitForm.bind(this);
    this.validateUsernameMaxLen = this.validateUsernameMaxLen.bind(this);
    this.onClickUsernameInput = this.onClickUsernameInput.bind(this);
    this.validateEmailFormat = this.validateEmailFormat.bind(this);
    this.onClickEmailInput = this.onClickEmailInput.bind(this);
    this.validatePasswordFormat = this.validatePasswordFormat.bind(this);
    this.onClickPasswordInput = this.onClickPasswordInput.bind(this);
    this.validateSamePassword = this.validateSamePassword.bind(this);
    this.onClickConfirmPasswordInput = this.onClickConfirmPasswordInput.bind(this);
  }

  onClickUsernameInput() {
    document.getElementById("username-input").className = "input active-input";
    document.getElementById("username-icon").src = "static/img/username_icon_active.png";
  }

  validateUsernameMaxLen() {
    let username = document.getElementById("username-value").value;
    let username_input = document.getElementById("username-input");
    let username_icon = document.getElementById("username-icon");
    let error_message = document.getElementById("error-message");
    username_input.className = "input";
    if (username.length == "") {
      username_input.className = "input wrong-input";
      error_message.innerHTML = "The username can't be empty.";
      error_message.className = "error-message show";
      username_icon.src = "static/img/username_icon_wrong.png";
      this.setState({ error_username: 1 });
    } else if (username.length > 150) {
      username_input.className = "input wrong-input";
      error_message.innerHTML = "The username can't be more than 150 characters.";
      error_message.className = "error-message show";
      username_icon.src = "static/img/username_icon_wrong.png";
      this.setState({ error_username: 1 });
    } else {
      username_input.className = "input";
      error_message.innerHTML = "";
      error_message.className = "error-message";
      this.setState({ error_username: 0 });
    }
  }

  onClickEmailInput() {
    document.getElementById("email-input").className = "input active-input";
    document.getElementById("email-icon").src =
      "static/img/email_icon_active.png";
  }

  validateEmailFormat() {
    let email = document.getElementById("email-value").value;
    let email_input = document.getElementById("email-input");
    let email_icon = document.getElementById("email-icon");
    let error_message = document.getElementById("error-message");
    let symbol = email.indexOf("@");
    let next = email.charAt(symbol + 1);
    email_input.className = "input";
    if (email.length == "") {
      email_input.className = "input wrong-input";
      error_message.innerHTML = "The email can't be empty.";
      error_message.className = "error-message show";
      email_icon.src = "static/img/email_icon_wrong.png";
      this.setState({ error_email: 1 });
    } else if (!email.includes("@")) {
      email_input.className = "input wrong-input";
      error_message.innerHTML = "The email should contain @ symbol.";
      error_message.className = "error-message show";
      email_icon.src = "static/img/email_icon_wrong.png";
      this.setState({ error_email: 1 });
    } else if (next == "") {
      email_input.className = "input wrong-input";
      error_message.innerHTML = "The email should contain the domain.";
      error_message.className = "error-message show";
      email_icon.src = "static/img/email_icon_wrong.png";
      this.setState({ error_email: 1 });
    } else {
      email_input.className = "input";
      error_message.innerHTML = "";
      error_message.className = "error-message";
      this.setState({ error_email: 0 });
    }
  }

  onClickPasswordInput() {
    document.getElementById("password-input").className = "input active-input";
    document.getElementById("password-icon").src =
      "static/img/password_icon_active.png";
  }

  validatePasswordFormat() {
    let password = document.getElementById("password-value").value;
    let password_input = document.getElementById("password-input");
    let password_icon = document.getElementById("password-icon");
    let error_message = document.getElementById("error-message");
    password_input.className = "input";
    if (password.length == "") {
      password_input.className = "input wrong-input";
      error_message.innerHTML = "The password can't be empty.";
      error_message.className = "error-message show";
      password_icon.src = "static/img/password_icon_wrong.png";
      this.setState({ error_password: 1 });
    } else if (password.length < 6) {
      password_input.className = "input wrong-input";
      error_message.innerHTML = "The password should be atleast 6 characters long.";
      error_message.className = "error-message show";
      password_icon.src = "static/img/password_icon_wrong.png";
      this.setState({ error_password: 1 });
    } else {
      password_input.className = "input";
      error_message.innerHTML = "";
      error_message.className = "error-message";
      this.setState({ error_password: 0 });
    }
  }

  onClickConfirmPasswordInput() {
    document.getElementById("confirm-password-input").className = "input active-input";
    document.getElementById("confirm-password-icon").src = "static/img/password_icon_active.png";
  }

  validateSamePassword() {
    let confirm_password = document.getElementById("confirm-password-value").value;
    let password = document.getElementById("password-value").value;
    let confirm_password_input = document.getElementById("confirm-password-input");
    let confirm_password_icon = document.getElementById("confirm-password-icon");
    let error_message = document.getElementById("error-message");
    confirm_password_input.className = "input";
    if (confirm_password != password) {
      confirm_password_input.className = "input wrong-input";
      error_message.innerHTML = "The passwords do not match.";
      error_message.className = "error-message show";
      confirm_password_icon.src = "static/img/password_icon_wrong.png";
      this.setState({ error_confirm_password: 1 });
    } else {
      confirm_password_input.className = "input";
      error_message.innerHTML = "";
      error_message.className = "error-message";
      this.setState({ error_confirm_password: 0 });
    }
  }

  submitForm(e) {
    e.preventDefault();
    let username = document.getElementById("username-value").value;
    let email = document.getElementById("email-value").value;
    let password = document.getElementById("password-value").value;
    password.replace(/[|&;$%@"<>()+,]/g, "");
    let confirm_password = document.getElementById("confirm-password-value").value;
    confirm_password.replace(/[|&;$%@"<>()+,]/g, "");
    let cleaned_password = password.trim();
    let cleaned_confirm_password = confirm_password.trim();
    let csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0].value;
    if (
      this.state.error_username == 0 &&
      this.state.error_email == 0 &&
      this.state.error_password == 0 &&
      this.state.error_confirm_password == 0
    ) {
      $.ajax({
        method: "POST",
        url: "/accounts/signup/",
        data: {
          csrfmiddlewaretoken: csrfmiddlewaretoken,
          username: username,
          email: email,
          password1: cleaned_password,
          password2: cleaned_confirm_password
        },
        success: function(result) {
          if (result.location == "/") {
          	localStorage.setItem('email', email);
            window.location = "/";
          } else {
            let error_message = document.getElementById("error-message");
            error_message.innerHTML = "Network error";
            error_message.className = "error-message show";
          }
        },
        complete: function(xhr) {
          if (xhr.status == 400) {
            let error_message = document.getElementById("error-message");
            error_message.innerHTML =
              "An account with those details already exists.";
            error_message.className = "error-message show";
          }
        }
      });
    } else {
      let error_message = document.getElementById("error-message");
      error_message.innerHTML = "Error";
      error_message.className = "error-message show";
    }
  }
  render() {
    return (
      <div className="signup-container">
        <center>
          <div className="card">
            <div className="card-top">
              <div className="card-top-left">
                <img src="static/img/logo_without_bg.png" />
              </div>
              <div className="card-top-right">
                <img src="static/img/account.png" />
                <h1>Create your account</h1>
              </div>
            </div>
            <div className="form">
              <form onSubmit={this.submitForm}>
                <div className="input" id="username-input">
                  <img id="username-icon" src="static/img/username_icon.png" />
                  <input
                    onFocus={this.onClickUsernameInput}
                    onBlur={this.validateUsernameMaxLen}
                    className="field"
                    type="text"
                    pattern=".{1,150}"
                    name="username"
                    id="username-value"
                    placeholder="Username"
                    required
                  />
                </div>
                <div className="input" id="email-input">
                  <img id="email-icon" src="static/img/email_icon.png" />
                  <input
                    onFocus={this.onClickEmailInput}
                    onBlur={this.validateEmailFormat}
                    className="field"
                    type="email"
                    name="email"
                    id="email-value"
                    placeholder="Email"
                    required
                  />
                </div>
                <div className="input" id="password-input">
                  <img id="password-icon" src="static/img/password_icon.png" />
                  <input
                    onFocus={this.onClickPasswordInput}
                    onBlur={this.validatePasswordFormat}
                    className="field"
                    type="password"
                    pattern=".{6,}"
                    name="password"
                    id="password-value"
                    placeholder="Password"
                    required
                  />
                </div>
                <div className="input" id="confirm-password-input">
                  <img
                    id="confirm-password-icon"
                    src="static/img/password_icon.png"
                  />
                  <input
                    onFocus={this.onClickConfirmPasswordInput}
                    onBlur={this.validateSamePassword}
                    className="field"
                    type="password"
                    pattern=".{6,}"
                    name="confirmpassword"
                    id="confirm-password-value"
                    placeholder="Confirm Password"
                    required
                  />
                </div>
                <DjangoCSRFToken />
                <p className="error-message" id="error-message" />
                <input type="submit" name="submit" value="sign up" />
              </form>
            </div>
            <a className="login-link" href="/accounts/login/">
              Already have an account?
            </a>
          </div>
        </center>
      </div>
    );
  }
}

export default Signup;
