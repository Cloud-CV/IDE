
import React from 'react';
import { render } from 'react-dom';
import { Router, Route, hashHistory } from 'react-router';
import App from './app.js';
import Signup from './signup.js';
import '../css/style.css';

render(
  <Router history={hashHistory}>
    <Route path="/" component={App} />
    <Route path="/signup" component={Signup} />
  </Router>, document.getElementById('app')
);
