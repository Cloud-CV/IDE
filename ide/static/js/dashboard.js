import React from 'react';
import Modal from 'react-modal';
import Card from './card';

const infoStyle = {
  content : {
    top                   : '50%',
    left                  : '55%',
    right                 : '60%',
    bottom                : 'auto',
    marginRight           : '-50%',
    transform             : 'translate(-50%, -50%)',
    borderRadius          : '8px'
  },
  overlay: {
    zIndex                : 100
  }
};

class Dashboard extends React.Component{
  constructor(props) {
    super(props);
    this.state = {modalIsOpen: false}
    this.getModelData = this.getModelData.bind(this);
    this.deleteModel = this.deleteModel.bind(this);
    this.openModal = this.openModal.bind(this);
    this.closeModal = this.closeModal.bind(this);
    this.modalContent = null;
    this.modalHeader = null;
  }

  getModelData(){
    var userid = localStorage.getItem("userID");
        $.ajax({
          url: '/getModel',
          dataType: 'json',
          type: 'POST',
          data: {userID: userid},
          success : function (response) {
            if (response.result == 'success') {
              var obj = JSON.stringify(response.data);
              localStorage.setItem("obj",obj);
              return obj;
            }
            else if (response.result == 'error') {
              if (localStorage.hasOwnProperty('obj')) {
                localStorage.removeItem("obj");
              }
            }
            return response.data;
          }.bind(this),
          error() {
          }
        });
  }

  deleteModel(model_id){
    var userid = localStorage.getItem("userID");
        $.ajax({
          url: '/deleteModel',
          type: 'POST',
          dataType: 'json',
          data: {userID: userid, modelid: model_id},
          success : function (response) {
            if (response.result == 'success') {
              this.modalContent = response.data;
              this.openModal();
            }
          }.bind(this),
          error() {
          }
        });
  }

  openModal() {
    this.setState({ modalIsOpen: true });
  }

  closeModal() {
    this.setState({ modalIsOpen: false });
    location.reload(true);
  }

  componentDidMount() {
    this.getModelData();
  }

  render() {
      if (localStorage.hasOwnProperty('userID')) {
      if (!localStorage.hasOwnProperty('obj')) {
        elements=<div><p style={{fontStyle: 'italic', marginTop: '2em'}}>No models found</p></div>;
      }
      else {
      var data_array = JSON.parse(localStorage.getItem("obj"));
      var len = Object.keys(data_array).length/2;
      var elements=[];
      for (var i = 1; i < len+1; i++) {
        elements.push(<Card ModelFunction={ this.deleteModel } ModelName={ data_array["Model"+i+"_Name"] } ModelID={ data_array["Model"+i+"_ID"] } />)
      }
      }
      return (
      <center>
        <div className="cont" style={{width: '80%', marginTop: '2em', display: 'inline-block'}}>
          <h1 style={{color: '#455062', fontSize: '2em', textAlign: 'left', fontWeight: 'bolder'}}>DASHBOARD</h1>
          <br />
          <p style={{textAlign: 'left'}}>
          <a href="#" style={{
            textDecoration: 'none', 
            color: '#fff', 
            fontWeight: 'bold', 
            background: '#222f3e', 
            fontSize: '0.7em', 
            padding: 15, 
            paddingRight: 20, 
            paddingLeft: 20, 
            borderRadius: 200
          }}>
          <span className="glyphicon glyphicon-plus" aria-hidden="true"></span> &nbsp;CREATE NEW MODEL</a>
          </p>
          <br />
          <div className="all-cards" style={{marginLeft: '-2.6em', display: 'inline-block'}}>
          {elements}
          </div>
        </div>
        <Modal
          isOpen={this.state.modalIsOpen}
          onRequestClose={this.closeModal}
          style={infoStyle}
          contentLabel="Modal">
          <button type="button" style={{padding: 5+'px'}} className="close" onClick={this.closeModal}>&times;</button>
          <h4>{ this.modalHeader }</h4>
          { this.modalContent }
        </Modal>
      </center>
      );
      }
      else {
        window.open("#","_self");
        return null;
      }
  }
}

export default Dashboard;
