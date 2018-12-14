import React from "react";
import { findDOMNode } from "react-dom";
import ReactTooltip from "react-tooltip";

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.scrollToAbout = this.scrollToAbout.bind(this);
    this.scrollToFeatures = this.scrollToFeatures.bind(this);
    this.scrollToTeam = this.scrollToTeam.bind(this);
    this.showFirstRow = this.showFirstRow.bind(this);
    this.showSecondRow = this.showSecondRow.bind(this);
  }

  scrollToAbout() {
    this.refs.about.scrollIntoView({ behavior: "smooth" });
  }

  scrollToFeatures() {
    this.refs.features.scrollIntoView({ behavior: "smooth" });
  }

  scrollToTeam() {
    this.refs.team.scrollIntoView({ behavior: "smooth" });
  }

  showFirstRow() {
    let slider = document.getElementsByClassName("slider")[0];
    slider.style.left = "0";
    let dot1 = document.getElementsByClassName("dot1")[0];
    let dot2 = document.getElementsByClassName("dot2")[0];
    let dot3 = document.getElementsByClassName("dot3")[0];
    dot1.className = "dot1 active";
    dot2.className = "dot2";
    dot3.className = "dot3";
  }

  showSecondRow() {
    let slider = document.getElementsByClassName("slider")[0];
    slider.style.left = "-100%";
    let dot1 = document.getElementsByClassName("dot1")[0];
    let dot2 = document.getElementsByClassName("dot2")[0];
    let dot3 = document.getElementsByClassName("dot3")[0];
    dot1.className = "dot1";
    dot2.className = "dot2 active";
    dot3.className = "dot3";
  }

  render() {
    return (
      <div>
        <div className="home-container">
          <div className="header">
            <img src="static/img/logo_without_bg.png" className="logo" />
            <div className="links">
              <a href="/accounts/signup/" id="signup">
                Sign up
              </a>
              <a onClick={() => this.scrollToTeam()}>Team</a>
              <a onClick={() => this.scrollToFeatures()}>Features</a>
              <a onClick={() => this.scrollToAbout()}>About</a>
            </div>
            <div className="clear" />
          </div>
          <div className="first">
            <div className="first-left">
              <h1>Fabrik</h1>
              <p>
                Collaboratively build, visualize <br /> and design neural nets
                in browser.
              </p>
              <br />
              <a href="/" className="try">
                Try it now!
              </a>
            </div>
            <div className="first-right">
              <img src="static/img/graphic1.png" />
            </div>
            <div className="clear" />
          </div>
          <div className="second" ref="about">
            <h1>About</h1>
            <center>
              <div className="large-box">
                <br />
                <p>
                  <b>Fabrik</b> is an online collaborative platform to build,
                  visualize and train deep learning models via a simple
                  drag-and-drop interface. It allows researchers to collectively
                  develop and debug models using a web GUI that supports
                  importing, editing and exporting networks to popular
                  frameworks like Caffe, Keras, and TensorFlow.
                </p>
              </div>
            </center>
          </div>
          <div className="third" ref="features">
            <br />
            <h1>Features</h1>
            <center>
              <div className="features">
                <div className="slider" id="slider">
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(
                        findDOMNode(this.refs.load_model_from_zoo_text)
                      );
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(
                        findDOMNode(this.refs.load_model_from_zoo_text)
                      );
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="load_model_from_zoo_text"
                      data-tip="You can choose a model from a variety of models located in the model zoo belonging to different categories such as Recognition, Detection, Retrieval, Seq2Seq, Captioning and Segmentation."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/folder.png" />
                    <p>Load Model From Zoo</p>
                    <div className="line" id="line_model_zoo" />
                  </div>
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(
                        findDOMNode(this.refs.load_model_from_input)
                      );
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(
                        findDOMNode(this.refs.load_model_from_input)
                      );
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="load_model_from_input"
                      data-tip="You can choose a model from a variety of models located in the model zoo belonging to different categories such as Recognition, Detection, Retrieval, Seq2Seq, Captioning and Segmentation."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/input.png" />
                    <p>Load Model From Input</p>
                    <div className="line" id="line_load_model_from_input" />
                  </div>
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(findDOMNode(this.refs.build_model));
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(findDOMNode(this.refs.build_model));
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="build_model"
                      data-tip="There are many built-in components called layers which can be used to build a neural net by simply clicking them to add them into the canvas. The layers are grouped into categories such as Data, Vision, Recurrent, Utitlity, Activation/Neuron, Normalization, Common, Noise, Loss and Wrapper."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/build.png" />
                    <p>Build Model</p>
                    <div className="line" id="line_build_model" />
                  </div>
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(findDOMNode(this.refs.export_model));
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(findDOMNode(this.refs.export_model));
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="export_model"
                      data-tip="You can export your model in any of the three frameworks. On exporting your model, the model file is downloaded into your local system which you can use to import your model into the canvas."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/export.png" />
                    <p>Export Model</p>
                    <div className="line" id="line_export_model" />
                  </div>
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(findDOMNode(this.refs.import_model));
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(findDOMNode(this.refs.import_model));
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="import_model"
                      data-tip="You can select your model file from your local system to load it into the canvas and then you can edit it and also perform other actions like share, export."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/import.png" />
                    <p>Import Model</p>
                    <div className="line" id="line_import_model" />
                  </div>
                  <div
                    onMouseOver={() => {
                      ReactTooltip.show(
                        findDOMNode(this.refs.real_time_collaboration)
                      );
                    }}
                    onMouseOut={() => {
                      ReactTooltip.hide(
                        findDOMNode(this.refs.real_time_collaboration)
                      );
                    }}
                    className="box"
                  >
                    {/* eslint-disable max-len */}
                    <p
                      className="tooltip-txt"
                      ref="real_time_collaboration"
                      data-tip="You can work collaboratively with other people, with whom the model is shared, in real time. All the edits that are made will be reflected to everyone who are working on the model."
                    />
                    {/* eslint-enable max-len */}
                    <ReactTooltip />
                    <img src="static/img/collab.png" />
                    <p>Real Time Collaboration</p>
                    <div className="line" id="line_real_time_collaboration" />
                  </div>
                </div>
                <div className="dots">
                  <a onClick={() => this.showFirstRow()}>
                    <div className="dot1 active" />
                  </a>
                  <a onClick={() => this.showSecondRow()}>
                    <div className="dot2" />
                  </a>
                </div>
              </div>
            </center>
            <br />
          </div>
          <div className="fourth" ref="team">
            <h1>Team</h1>
            <center>
              <div className="members">
                <br />
                <div className="member">
                  <img src="static/img/Deshraj.png" />
                  <p>Deshraj</p>
                  <p className="social-icons">
                    <a href="http://deshraj.xyz">
                      <span
                        className="glyphicon glyphicon-user"
                        aria-hidden="true"
                      />
                    </a>
                    <a href="mailto:deshrajdry@gmail.com">
                      <span
                        className="glyphicon glyphicon-envelope"
                        aria-hidden="true"
                      />
                    </a>
                  </p>
                </div>
                <div className="member">
                  <img src="static/img/Ram_Ramrakhya.jpg" />
                  <p>Ram Ramrakhya</p>
                  <p className="social-icons">
                    <a href="http://github.com/Ram81">
                      <span
                        className="glyphicon glyphicon-user"
                        aria-hidden="true"
                      />
                    </a>
                    <a href="mailto:ramramrakhya81@gmail.com">
                      <span
                        className="glyphicon glyphicon-envelope"
                        aria-hidden="true"
                      />
                    </a>
                  </p>
                </div>
                <div className="member">
                  <img src="static/img/user 2.png" />
                  <p>Utsav Garg</p>
                  <p className="social-icons">
                    <a href="#">
                      <span
                        className="glyphicon glyphicon-user"
                        aria-hidden="true"
                      />
                    </a>
                    <a href="#">
                      <span
                        className="glyphicon glyphicon-envelope"
                        aria-hidden="true"
                      />
                    </a>
                  </p>
                </div>
                <div className="member">
                  <img src="static/img/Viraj_Prabhu.png" />
                  <p>Viraj Prabhu</p>
                  <p className="social-icons">
                    <a href="http://virajprabhu.github.io">
                      <span
                        className="glyphicon glyphicon-user"
                        aria-hidden="true"
                      />
                    </a>
                    <a href="mailto:virajp@gatech.edu">
                      <span
                        className="glyphicon glyphicon-envelope"
                        aria-hidden="true"
                      />
                    </a>
                  </p>
                </div>
              </div>
            </center>
          </div>
          <div className="footer">
            <br />
            <center>
              <div className="footer-content">
                <div className="footer-content-left">
                  <p>Fabrik</p>
                  <p className="sub-txt">
                    Collaboratively build, visualize <br /> and design neural
                    nets in browser
                  </p>
                  <p className="sub-txt">
                    <span>©</span> CloudCV
                  </p>
                </div>
                <div className="footer-content-right">
                  <div className="icons">
                    <a href="https://github.com/Cloud-CV/Fabrik">
                      <img src="static/img/github.png" />
                    </a>
                    <a href="https://www.facebook.com/mlcloudcv">
                      <img src="static/img/facebook.png" />
                    </a>
                    <a href="https://www.linkedin.com/company/cloudcv">
                      <img src="static/img/linkedin.png" />
                    </a>
                    <a href="https://twitter.com/project_cloudcv">
                      <img src="static/img/twitter.png" />
                    </a>
                  </div>
                </div>
                <div className="clear" />
              </div>
            </center>
          </div>
        </div>
      </div>
    );
  }
}

export default Home;
