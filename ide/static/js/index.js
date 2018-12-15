import React from 'react';
import { render } from 'react-dom';
import { Router, Route, hashHistory } from 'react-router';
import App from './app.js';
import Dashboard from './dashboard.js';
import '../css/style.css';

render(
  <Router history={hashHistory}>
    <Route path="/" component={App} />
    <Route path="/dashboard" component={Dashboard} />
  </Router>, document.getElementById('app')
);
