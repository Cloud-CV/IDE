import React from 'react';

class ModelElement extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    if (this.props.children){
      return (
        <div id={this.props.displayName} className="col-sm-6 col-md-4">
          <div className="btn thumbnail text-center" onClick={() => this.props.importNet('sample'+this.props.framework, this.props.id)}>
          <img src={"/static/img/zoo/" + this.props.id + ".png"} />
            <div className="caption">
              <h3>{this.props.displayName}</h3>
            </div>
          </div>
        </div>
        // <a
          // style={{color: "#848a92"}}
          // className="btn"
          // onClick={() => this.props.importNet('sample'+this.props.framework, this.props.id)}
          // >
          // {this.props.children}
        // </a>
      );
    }
    else
      return null
  }
}

ModelElement.propTypes = {
  importNet: React.PropTypes.func,
  framework: React.PropTypes.string.isRequired,
  children: React.PropTypes.string.isRequired,
  id: React.PropTypes.string.isRequired,
  displayName: React.PropTypes.string.isRequired
};

export default ModelElement;
