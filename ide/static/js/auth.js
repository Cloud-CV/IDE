$(".dropdown-trigger").dropdown();
$(".sidenav").sidenav();

$("#id_password2").one("focusin", function(e) {
  $("#id_password1, #id_password2").on("keyup", function(ev) {
    var samePasswords = $("#id_password1").val() == $("#id_password2").val();
    $("#id_password2").toggleClass("valid", samePasswords).toggleClass("invalid", !samePasswords);
  });
});