import React from 'react';
import Content from './content';
import Login from './loginPage';

export default function () { // eslint-disable-line
  if (location.pathname == '/login/') {
    return (
      <div className="app">
        <Login />
      </div>
    );
  } else {
    return (
      <div className="app">
        <Content />
      </div>
    );
  }
}
