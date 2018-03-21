import React from 'react';
import ReactTooltip from 'react-tooltip';
import data from './data';

class TopBar extends React.Component {
  render() {
    //Delete button class options
    var delete_button_class_options = "btn btn-default";
    if (this.props.selectedLayer){
      const layer = this.props.net[this.props.selectedLayer];
      if ((layer.info.phase === null) && (this.props.selectedPhase === 1) && (data[layer.info.type].learn))
        delete_button_class_options = "btn btn-default disabled";
    }

    //Delete button's span's class class_options
    var delete_span_class_options = null;
    if (!this.props.deleteMode)
      delete_span_class_options = "	glyphicon glyphicon-remove";
    else
      delete_span_class_options = "	glyphicon glyphicon-remove text-warning";


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
                  <button id="topbar-icon" className={delete_button_class_options}
                  onClick={() => this.props.toggleDeleteMode()} data-tip="Select a layer to delete it">
                    <span className={delete_span_class_options} aria-hidden="true"></span>
                  </button>
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
  deleteMode: React.PropTypes.bool,
  toggleDeleteMode: React.PropTypes.func
};

export default TopBar;
