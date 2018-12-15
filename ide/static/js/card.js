import React from 'react';
import '../css/dash_style.css';

class Card extends React.Component {
  render() {
    return(
            <div style={{display: 'inline-block'}}>
              <div className="card" style={{
                background: '#fff', 
                height: 290, 
                width: 240, 
                borderRadius: 20, 
                display: 'inline-block', 
                margin: 20, 
                boxShadow: '0 2px 6px rgba(112,112,112,0.2)'
              }}>
              <img style={{marginTop: '4.5em'}} src="static/img/thumb.png" height="60%;" />
              </div>
              <div className="overlay">
                <br />
                <div className="social" style={{marginTop: '5em'}}>
                  <div style={{
                    margin: 10, 
                    display: 'inline-block', 
                    color: '#fff', 
                    fontFamily: '"Proxima Nova",sans-serif', 
                    width: 10, 
                    height: 10, 
                    border: '3px solid #fff', 
                    padding: 20, 
                    borderRadius: '50%'
                  }}>
                  <a href={"/load?id=" + this.props.ModelID} style={{textDecoration: 'none', color: '#fff'}}>
                  <p style={{marginTop: '-8px', marginLeft: '-6px'}}>
                  <span className="glyphicon glyphicon-pencil" aria-hidden="true"></span>
                  </p>
                  </a>
                  </div>
                  <div style={{margin: 10, 
                    display: 'inline-block', 
                    color: '#fff', 
                    fontFamily: '"Proxima Nova",sans-serif', 
                    width: 10, 
                    height: 10, 
                    border: '3px solid #fff', 
                    padding: 20, 
                    borderRadius: '50%'
                  }}>
                  <a onClick={
                    () => this.props.ModelFunction(this.props.ModelID)
                  } style={{
                    textDecoration: 'none', 
                    color: '#fff', 
                    cursor: 'pointer'
                  }}>
                  <p style={{
                    marginTop: '-8px', 
                    marginLeft: '-7px'
                  }}>
                  <span className="glyphicon glyphicon-trash" aria-hidden="true"></span>
                  </p>
                  </a>
                  </div>
                </div>
              </div>
              <h3 style={{marginTop: '-12em'}}>{this.props.ModelName}</h3>
            </div>
      );
  }
}

Card.propTypes = {
  ModelName: React.PropTypes.string,
  ModelID: React.PropTypes.number,
  ModelFunction: React.PropTypes.func
};


export default Card;
