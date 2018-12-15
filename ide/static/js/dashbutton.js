import React from 'react';

class DashButton extends React.Component {
	constructor(props) {
		super(props);
		this.openDash = this.openDash.bind(this);
	}
	openDash(){
		window.location.href = "/#/dashboard";
	}
	render(){
		return(
			<div>
			<h5 className="sidebar-heading" id="sidebar-dash-button" onClick={() => {this.openDash()}}>
			<div>DASHBOARD</div>
			</h5>
			</div>
			);
	}
}

export default DashButton;
