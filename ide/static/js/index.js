
import React from 'react';
import { render } from 'react-dom';
import { Router, Route, hashHistory } from 'react-router';
import App from './app.js';
import Home from './home.js';
import '../css/style.css';

render(
  <Router history={hashHistory}>
    <Route path="/" component={App} />
    <Route path="/home" component={Home} />
  </Router>, document.getElementById('app')
);
