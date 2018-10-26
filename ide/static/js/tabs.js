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
    $('.btn-mode-toggle').click(e => {
      if (document.body.className.search('dark') == -1) {
        document.body.className = 'app-dark';
        e.target.id = 'btn-dark';
        e.target.innerHTML = 'Light Mode';
      } else {
        document.body.className = '';
        e.target.id = 'btn-light';
        e.target.innerHTML = 'Dark Mode';
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
        <button className="btn-mode-toggle">
          Dark Mode
        </button>
      </div>
    );
  }
}

Tabs.propTypes = {
  changeNetPhase: React.PropTypes.func,
  selectedPhase: React.PropTypes.number
};

export default Tabs;
