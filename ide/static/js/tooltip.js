import React from 'react';
import TooltipData from './tooltipData';
import data from './data';
import ReactTooltip from 'react-tooltip'

class Tooltip extends React.Component {
  constructor(props) {
    super(props);
  }
 
  render() {
    if (this.props.hoveredLayer) {
      const params = [];
      const props = [];
      const layer = this.props.net[this.props.hoveredLayer];
      //console.log(layer);

      Object.keys(data[layer.info.type].params).forEach(param => {
        params.push(
          <TooltipData
            id={param}
            key={param}
            data={data[layer.info.type].params[param]}
            value={layer.params[param]}
            disabled={(layer.info.phase === null) && (this.props.selectedPhase === 1) && (data[layer.info.type].learn)}
            changeField={this.changeParams}
          />
        );
      });

      Object.keys(data[layer.info.type].props).forEach(prop => {
        props.push(
          <TooltipData
            id={prop}
            key={prop}
            data={data[layer.info.type].props[prop]}
            value={layer.props[prop]}
            disabled={(layer.info.phase === null) && (this.props.selectedPhase === 1) && (data[layer.info.type].learn)}
            changeField={this.changeProps}
          />
        );
      });

      return (
        <ReactTooltip id='getContent' effect='solid' place='right' class='customTooltip'>
          <div>
            <div>
              {props}
            </div>
            <br />
            <div>
              {params}
            </div>
          </div>
        </ReactTooltip>
    )
}
     else return null;
  }
}

Tooltip.propTypes = {
  hoveredLayer: React.PropTypes.string,
  net: React.PropTypes.object,
};

export default Tooltip;