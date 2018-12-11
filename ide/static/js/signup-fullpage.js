import React from "react";
import "../css/signup_style.css";

class SignUp extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            name: "",
            email: "",
            username: "",
            password: ""
        };
        this.onSubmit = this.onSubmit.bind(this);
        this.onChange = this.onChange.bind(this);
    }

    onSubmit(e) {
        e.preventDefault();
        console.log(this.state);
    }
    onChange(e) {
        this.setState({ [e.target.name]: e.target.value });
    }

    render() {
        return (
            <div className="body">
                <div className="container">
                    <div className="row">
                        <div className="col-sm-3" />
                        <div className="col-sm-6 content">
                            <img src="/static/img/logo.png" className="logo" />
                            <h1>Sign Up</h1>
                            <form onSubmit={this.onSubmit}>
                                <div className="form-group">
                                    <label htmlFor="inputName">Name</label>
                                    <input
                                        type="name"
                                        name="name"
                                        className="form-control"
                                        id="inputName"
                                        placeholder="Enter name"
                                        value={this.state.name}
                                        onChange={this.onChange}
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="inputEmail">Email</label>
                                    <input
                                        type="email"
                                        name="email"
                                        className="form-control"
                                        id="inputEmail"
                                        placeholder="Email"
                                        value={this.state.email}
                                        onChange={this.onChange}
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="inputUsername">
                                        Username
                                    </label>
                                    <input
                                        type="username"
                                        name="username"
                                        className="form-control"
                                        id="inputUsername"
                                        placeholder="Username"
                                        value={this.state.username}
                                        onChange={this.onChange}
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="inputPassword">
                                        Password
                                    </label>
                                    <input
                                        type="password"
                                        name="password"
                                        className="form-control"
                                        id="inputPassword"
                                        placeholder="Password"
                                        value={this.state.password}
                                        onChange={this.onChange}
                                    />
                                </div>
                                <div className="text-center">
                                    <button
                                        type="submit"
                                        className="btn btn-primary"
                                    >
                                        Sign Up
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default SignUp;
