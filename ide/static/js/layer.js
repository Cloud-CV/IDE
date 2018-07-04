import React from 'react';
import data from './data';

class Layer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      modalIsOpen: false,
      layerId: 0
    }
    this.onAddComment = this.onAddComment.bind(this);
  }
  componentDidMount() {
    instance.addLayerEndpoints(this.props.id,
      data[this.props.type].endpoint.src,
      data[this.props.type].endpoint.trg
    );
  }
  componentWillUnmount() {
    instance.deleteEndpoint(`${this.props.id}-s0`);
    instance.deleteEndpoint(`${this.props.id}-t0`);
  }
  onAddComment(event, layerId) {
    //console.log('ola')
    this.setState({ modalIsOpen: true, layerId: layerId })
  }
  render() {
    return (
      <div
        className={`layer ${this.props.class}`}
        id={this.props.id}
        style={{
          top:this.props.top,
          left:this.props.left,
          background: data[this.props.type].color
        }}
        data-type={this.props.type}
        onClick={(event) => this.props.click(event, this.props.id)}
        onMouseEnter={(event) => this.props.hover(event, this.props.id)}
        data-tip='tooltip'
        data-for='getContent'
      >
        <a style={{
          position: 'absolute',
          top: '-5px',
          right: '-2px',
          color: 'white'
        }} onClick={(event) => this.onAddComment(event, this.props.id)}>
          <span className="glyphicon glyphicon-plus-sign" style={{ fontSize: '15px', paddingRight: '5px'}} aria-hidden="true"></span>
        </a>
        {data[this.props.type].name}
      </div>
    );
  }
}

Layer.propTypes = {
  id: React.PropTypes.string.isRequired,
  type: React.PropTypes.string.isRequired,
  top: React.PropTypes.string.isRequired,
  left: React.PropTypes.string.isRequired,
  class: React.PropTypes.string,
  click: React.PropTypes.func,
  hover: React.PropTypes.func
};

export default Layer;
