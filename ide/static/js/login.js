import React from 'react';
import { GithubLoginButton, GoogleLoginButton } from 'react-social-login-buttons';

class Login extends React.Component {
  constructor(props) {
    super(props);
    this.checkLogin = this.checkLogin.bind(this);
    this.logoutUser = this.logoutUser.bind(this);
  }
  componentWillMount() {
    this.setState({ loginState: false });
    this.checkLogin();
  }
  checkLogin() {
    $.ajax({
      url: '/backendAPI/checkLogin',
      type: 'GET',
      processData: false,  // tell jQuery not to process the data
      contentType: false,
      success: function (response) {
        if (response.result) {
          this.setState({ loginState: response.result });
          this.props.setUserId(response.user_id);
        }
      }.bind(this),
      error: function () {
        this.setState({ loginState: false });
        this.addError("Error occurred while logging in");
      }.bind(this)
    });
  }
  logoutUser() {
    $.ajax({
      url: '/accounts/logout',
      type: 'GET',
      processData: false,  // tell jQuery not to process the data
      contentType: false,
      success: function (response) {
        if (response) {
          this.setState({ loginState: false });
          this.props.setUserId(null);
        }
      }.bind(this),
      error: function () {
        this.setState({ loginState: true });
        this.addError("Error occurred while logging out");
      }.bind(this)
    });
  }
  render() {
    if(this.state.loginState) {
      return (
        <div>
          <a className="btn btn-block extra-buttons text-left" onClick={ () => this.logoutUser() }>Logout</a>
        </div>
      )
    }
    else {
      return (
        <div>
          <GithubLoginButton text="Login with Github" onClick={() => window.location="/accounts/github/login"} />
          <GoogleLoginButton text="Login with Google" onClick={() => window.location="/accounts/google/login"} />
        </div>
      )
    }
  }
}

Login.propTypes = {
  setUserId: React.PropTypes.func
};

export default Login;
