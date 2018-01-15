import React from 'react';
import {GithubLoginButton, GoogleLoginButton} from 'react-social-login-buttons';
import Modal from 'react-modal';

class Login extends React.Component {
  constructor(props) {
    super(props);
    this.getMyModels = this.getMyModels.bind(this);
  }
  getMyModels() {
    $.ajax({
      url: '/backendAPI/getModels',
      type: 'GET',
      processData: false,  // tell jQuery not to process the data
      contentType: false,
      success: function (response) {
        this.setState({ "myModels": response })
        var toRet = []
        this.state.myModels.forEach(function (el) {
          toRet.push(
            <a
              key={el.id}
              className="btn my-models-list-item"
              onClick={() => {
                this.props.loadDb(el.id)
                this.closeModal()
              }}
            >
              {el.name}
            </a>
          )
        }.bind(this))
        this.setState({ "models": toRet })
        this.forceUpdate()
      }.bind(this)
    });
  }

  componentWillMount() {
    this.setState({ "loginState": "in" })
    this.setState({"models": ""})
    this.getMyModels()
    this.checkLogin();
  }
  checkLogin() {
    $.ajax({
      url: '/backendAPI/checkLogin',
      type: 'GET',
      processData: false,  // tell jQuery not to process the data
      contentType: false,
      success: function (response) {
          this.setState({ "loginState": response.result })
          this.forceUpdate()
      }.bind(this)
    });
  }
  myModelLook() {
    {
      this.modalHeader = 'My models:';
      this.modalContent = (
      <div className="my-models-list-container">
        {this.state.models}
      </div>
    );
      this.openModal();
    }
  }
  openModal() {
    this.setState({"modalIsOpen": true})
  }
  closeModal() {
    this.setState({"modalIsOpen": false})
  }
  render() {
    const infoStyle = {
      content : {
        top                   : '50%',
        left                  : '50%',
        right                 : '60%',
        bottom                : 'auto',
        marginRight           : '-50%',
        transform             : 'translate(-50%, -50%)',
        borderRadius          : '8px',
        zIndex                : '200'
      },
      overlay: {
        zIndex                : 100
      }
    };
    if(this.state.loginState) {
      return (
        <div>
        <a href="/accounts/logout"><h5 className="zoo-modal-text">Logout</h5></a>
        <h5 className="zoo-modal-text" onClick={() => this.myModelLook() }>My Models</h5>
        <Modal
            isOpen={this.state.modalIsOpen}
            onRequestClose={this.closeModal}
            contentLabel="Modal"
            infoStyle={infoStyle}
            >
            
            <button type="button" style={{padding: 5+'px'}} className="close" onClick={this.closeModal}>&times;</button>
            <h4>{ this.modalHeader }</h4>
            { this.modalContent }
          </Modal>
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
  loadDb: React.PropTypes.func
};

export default Login;
