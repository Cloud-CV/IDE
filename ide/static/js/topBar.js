import React from 'react';
import ReactTooltip from 'react-tooltip';
import data from './data';

class TopBar extends React.Component {
  render() {
    var delete_button_options = null;
    var delete_button_data_tip = "Select a layer first"
    if (this.props.selectedLayer) {
      const layer = this.props.net[this.props.selectedLayer];
      var class_options = "btn";
      if ((layer.info.phase === null) && (this.props.selectedPhase === 1) && (data[layer.info.type].learn))
        class_options = "btn disabled";
      delete_button_options = (<ul className="dropdown-menu dropdown-menu-right">
        <li><a className={class_options} href="#" onClick={() => this.props.store_list_layers(this.props.selectedLayer)}>
        Add selected layer to 'delete' queue</a></li>
        <li><a className={class_options} href="#" onClick={() => this.props.delete_selected_layers()}>
        Empty 'delete' queue</a></li>
      </ul>);
      delete_button_data_tip = "";
    }

    return (
      <div className="topBar">
        <div className="topbar-row">
            <div className="topbar-col">
              <div className="form-group">
                  <div className="dropdown">
                    <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown"
                    onClick={() => this.props.zooModal()} data-tip="Load from zoo">
                      <span className="glyphicon glyphicon-folder-open" aria-hidden="true"></span>
                    </button>
                  </div>
              </div>
            </div>
            <div className="topbar-col">
              <div className="form-group">
                  <div className="dropdown">
                    <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown"
                    onClick={() => this.props.textboxModal()} data-tip="Load from input">
                      <span className="glyphicon glyphicon-align-left" aria-hidden="true"></span>
                    </button>
                  </div>
              </div>
            </div>
            <div className="topbar-col">
              <div className="form-group">
                <div className="dropdown">
                  <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown" data-tip="Export">
                    <span className="glyphicon glyphicon-export" aria-hidden="true"></span>
                  </button>
                  <ul className="dropdown-menu">
                    <li><a className="btn" href="#" onClick={() => this.props.exportNet('caffe')}>Caffe</a></li>
                    <li><a className="btn" href="#" onClick={() => this.props.exportNet('keras')}>Keras</a></li>
                    <li><a className="btn" href="#" onClick={() => this.props.exportNet('tensorflow')}>Tensorflow</a></li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="topbar-col">
              <div className="form-group">
                <div className="dropdown">
                  <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown" data-tip="Import">
                    <span className="glyphicon glyphicon-import" aria-hidden="true"></span>
                  </button>
                  <ul className="dropdown-menu">
                    <li>
                        <a className="btn">
                        <label htmlFor="inputFilecaffe">Caffe</label>
                        <input id="inputFilecaffe" type="file" accept=".prototxt" onChange={() => this.props.importNet('caffe', '')}/>
                        </a>
                    </li>
                    <li>
                        <a className="btn">
                        <label htmlFor="inputFilekeras">Keras</label>
                        <input id="inputFilekeras" type="file" accept=".json" onChange={() => this.props.importNet('keras', '')}/>
                        </a>
                    </li>
                    <li>
                        <a className="btn">
                        <label htmlFor="inputFiletensorflow">Tensorflow</label>
                        <input id="inputFiletensorflow" type="file" accept=".pbtxt" onChange={() => this.props.importNet('tensorflow', '')}/>
                        </a>
                    </li>
                    <li><a className="btn" onClick={() => this.props.urlModal()}>URL</a></li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="topbar-col">
              <div className="form-group">
                <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown"
                onClick={() => this.props.saveDb()} data-tip="Share">
                    <span className="glyphicon glyphicon-share" aria-hidden="true"></span>
                </button>
              </div>
            </div>
            <div className="topbar-col">
              <div className="form-group">
                <div className="dropdown">
                  <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control float-left"  data-toggle="dropdown"
                   data-tip={delete_button_data_tip}>
                    <span className="	glyphicon glyphicon-remove" aria-hidden="true"></span>
                  </button>
                  {delete_button_options}
                </div>
              </div>
            </div>
        </div>
      <ReactTooltip type="dark" multiline={true}/>
      </div>
    );
  }
}

TopBar.propTypes = {
  selectedLayer: React.PropTypes.string,
  net: React.PropTypes.object,
  selectedPhase: React.PropTypes.number,
  exportNet: React.PropTypes.func,
  importNet: React.PropTypes.func,
  saveDb: React.PropTypes.func,
  loadDb: React.PropTypes.func,
  zooModal: React.PropTypes.func,
  textboxModal: React.PropTypes.func,
  urlModal: React.PropTypes.func,
  store_list_layers: React.PropTypes.func,
  delete_selected_layers: React.PropTypes.func
};

export default TopBar;
