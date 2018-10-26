import React from 'react';

class Tabs extends React.Component {
  componentDidMount() {
    $('#phaseTabs button').click(e => {
      e.preventDefault();
      if (e.target.id === 'train') {
        this.props.changeNetPhase(0);
      } else if (e.target.id === 'test') {
        this.props.changeNetPhase(1);
      }
    });
    $('#sidebar-scroll')[0].scrollLeft = 100;
    $('.btn-mode-toggle').click(() => {
      if (document.body.className.search('dark') == -1) {
        document.body.className = 'app-dark';
      } else {
        document.body.className = '';
      }
    })
  }
  render() {
    let trainClass = 'btn-primary',
      testClass = 'btn-default';
    if (this.props.selectedPhase === 0) {
      trainClass = 'btn-primary';
      testClass = 'btn-default';
    } else if (this.props.selectedPhase === 1) {
      trainClass = 'btn-default';
      testClass = 'btn-primary';
    }
    return (
      <div>
        <li className="btn-group" role="group" id="phaseTabs">
          <button type="button" id="train" className={"btn "+trainClass}>Train</button>
          <button type="button" id="test" className={"btn "+testClass}>Test</button>
        </li>
        <div className="mode-toggle">
          <div className="sidebar-heading dark-mode-title">
            DARK MODE
          </div>
          <div className="btn-mode-toggle">
            <div id="toggle-circle"></div>
          </div>
        </div>
      </div>
    );
  }
}

Tabs.propTypes = {
  changeNetPhase: React.PropTypes.func,
  selectedPhase: React.PropTypes.number
};

export default Tabs;
